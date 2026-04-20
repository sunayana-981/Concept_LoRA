"""
Decomposed LoRA (DoRA) fine-tuning for CLIP.

DoRA (Liu et al. 2024 - "DoRA: Weight-Decomposed Low-Rank Adaptation") decomposes
the pretrained weight W₀ into magnitude m and direction V, then adapts only the
direction via standard LoRA while keeping m trainable:

    W_adapted = m * (W₀ + B@A) / ‖W₀ + B@A‖_col

where m ∈ ℝ^d_out is initialized to ‖W₀‖_col so that W_adapted ≈ W₀ at init.
"""

import os
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import CLIP_LoRA.clip as clip
from CLIP_LoRA.utils import cls_acc, pre_load_features
from CLIP_LoRA.datasets import build_dataset
from CLIP_LoRA.datasets.utils import build_data_loader
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Layer position indices (same as loralib/utils.py)
# ---------------------------------------------------------------------------

INDEX_POSITIONS_TEXT = {
    'top1':        [11],
    'top2':        [10, 11],
    'top3':        [9, 10, 11],
    'bottom':      [0, 1, 2, 3],
    'mid':         [4, 5, 6, 7],
    'up':          [8, 9, 10, 11],
    'half-up':     [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all':         list(range(12)),
}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top':         [11],
        'top3':        [9, 10, 11],
        'bottom':      [0, 1, 2, 3],
        'mid':         [4, 5, 6, 7],
        'up':          [8, 9, 10, 11],
        'half-up':     [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all':         list(range(12)),
    },
    'ViT-B/32': {
        'bottom':      [0, 1, 2, 3],
        'mid':         [4, 5, 6, 7],
        'up':          [8, 9, 10, 11],
        'half-up':     [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all':         list(range(12)),
    },
    'ViT-L/14': {
        'half-up':     list(range(12, 24)),
        'half-bottom': list(range(12)),
        'all':         list(range(24)),
    },
}


# ---------------------------------------------------------------------------
# DoRA Linear Layer
# ---------------------------------------------------------------------------

class LinearDoRA(nn.Module):
    """
    Replaces nn.Linear with a DoRA-adapted version.

    Forward: W_eff = (m / ‖W₀ + s·B@A‖_col) * (W₀ + s·B@A)
    - m: trainable magnitude vector (d_out,), init = ‖W₀‖_col
    - A: LoRA down-projection (r, d_in), init Kaiming uniform
    - B: LoRA up-projection (d_out, r), init zeros
    - s = lora_alpha / sqrt(r)  (rsLoRA-style scaling)
    """

    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        d_out, d_in = existing_linear.weight.shape
        self.r = r
        self.scaling = lora_alpha / math.sqrt(r)

        # Frozen pretrained weight
        self.weight = nn.Parameter(existing_linear.weight.data.clone(), requires_grad=False)
        if existing_linear.bias is not None:
            self.bias = nn.Parameter(existing_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # DoRA magnitude: initialize to column-wise L2 norm of pretrained weight
        col_norms = existing_linear.weight.data.norm(dim=1)  # (d_out,)
        self.magnitude = nn.Parameter(col_norms.clone())

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.scaling * (self.lora_B @ self.lora_A)   # (d_out, d_in)
        adapted = self.weight + delta_w                          # (d_out, d_in)

        col_norms = adapted.norm(dim=1, keepdim=True)            # (d_out, 1)
        weight_dora = (self.magnitude.unsqueeze(1) / col_norms) * adapted

        return F.linear(self.dropout(x), weight_dora, self.bias)


# ---------------------------------------------------------------------------
# DoRA Multi-head Attention wrapper
# ---------------------------------------------------------------------------

class MultiheadAttentionDoRA(nn.Module):
    """
    Wraps CLIP's nn.MultiheadAttention, replacing selected projections with DoRA.

    enable_lora: list of projections to adapt, subset of ['q', 'k', 'v', 'o']
    """

    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: list = ['q', 'k', 'v', 'o'],
        r: int = 4,
        lora_alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = existing_mha.embed_dim
        self.num_heads = existing_mha.num_heads
        self.head_dim  = existing_mha.head_dim
        self.batch_first = existing_mha.batch_first
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.dropout_p = 0.0  # handled externally

        has_bias = existing_mha.in_proj_bias is not None
        W = existing_mha.in_proj_weight.data      # (3*embed, embed)
        b = existing_mha.in_proj_bias.data if has_bias else None
        d = self.embed_dim

        # Build plain nn.Linear projections from the fused in_proj_weight
        def _make_linear(w_slice, b_slice):
            lin = nn.Linear(d, d, bias=has_bias)
            lin.weight.data.copy_(w_slice)
            if has_bias:
                lin.bias.data.copy_(b_slice)
            return lin

        q_lin = _make_linear(W[:d],    b[:d]    if has_bias else None)
        k_lin = _make_linear(W[d:2*d], b[d:2*d] if has_bias else None)
        v_lin = _make_linear(W[2*d:],  b[2*d:]  if has_bias else None)

        o_lin = nn.Linear(d, d, bias=existing_mha.out_proj.bias is not None)
        o_lin.weight.data.copy_(existing_mha.out_proj.weight.data)
        if existing_mha.out_proj.bias is not None:
            o_lin.bias.data.copy_(existing_mha.out_proj.bias.data)

        # Replace selected projections with DoRA
        dora_kwargs = dict(r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)
        self.q_proj = LinearDoRA(q_lin, **dora_kwargs) if 'q' in enable_lora else q_lin
        self.k_proj = LinearDoRA(k_lin, **dora_kwargs) if 'k' in enable_lora else k_lin
        self.v_proj = LinearDoRA(v_lin, **dora_kwargs) if 'v' in enable_lora else v_lin
        self.proj   = LinearDoRA(o_lin, **dora_kwargs) if 'o' in enable_lora else o_lin

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                pass
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None


# ---------------------------------------------------------------------------
# Apply / Save / Load DoRA
# ---------------------------------------------------------------------------

def apply_dora(args, clip_model):
    """Replace MHA modules with MultiheadAttentionDoRA and return the list."""
    dora_layers = []
    dora_kwargs = dict(
        enable_lora=args.params,
        r=args.r,
        lora_alpha=args.alpha,
        dropout_rate=args.dropout_rate,
    )

    if args.encoder in ('text', 'both'):
        indices = INDEX_POSITIONS_TEXT[args.position]
        for i, block in enumerate(clip_model.transformer.resblocks):
            if i not in indices:
                continue
            for name, submodule in list(block.named_children()):
                if isinstance(submodule, nn.MultiheadAttention):
                    new_attn = MultiheadAttentionDoRA(submodule, **dora_kwargs)
                    setattr(block, name, new_attn)
                    dora_layers.append(new_attn)

    if args.encoder in ('vision', 'both'):
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        for i, block in enumerate(clip_model.visual.transformer.resblocks):
            if i not in indices:
                continue
            for name, submodule in list(block.named_children()):
                if isinstance(submodule, nn.MultiheadAttention):
                    new_attn = MultiheadAttentionDoRA(submodule, **dora_kwargs)
                    setattr(block, name, new_attn)
                    dora_layers.append(new_attn)

    return dora_layers


def mark_only_dora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except DoRA-specific ones (lora_A, lora_B, magnitude)."""
    dora_param_names = {'lora_A', 'lora_B', 'magnitude'}
    for name, param in model.named_parameters():
        leaf = name.split('.')[-1]
        param.requires_grad = leaf in dora_param_names


def get_dora_parameters(model: nn.Module):
    dora_param_names = {'lora_A', 'lora_B', 'magnitude'}
    return [p for n, p in model.named_parameters() if n.split('.')[-1] in dora_param_names]


def _proj_state(proj):
    """Return (lora_A, lora_B, magnitude) tensors if DoRA, else None."""
    if isinstance(proj, LinearDoRA):
        return {
            'lora_A':    proj.lora_A.data.cpu(),
            'lora_B':    proj.lora_B.data.cpu(),
            'magnitude': proj.magnitude.data.cpu(),
        }
    return None


def save_dora(args, dora_layers):
    weights = {}
    for i, layer in enumerate(dora_layers):
        lw = {}
        for proj_key, proj_attr in [('q_proj', 'q_proj'), ('k_proj', 'k_proj'),
                                     ('v_proj', 'v_proj'), ('proj', 'proj')]:
            state = _proj_state(getattr(layer, proj_attr))
            if state is not None:
                lw[proj_key] = state
        weights[f'layer_{i}'] = lw

    metadata = {
        'r':        args.r,
        'alpha':    args.alpha,
        'encoder':  args.encoder,
        'params':   args.params,
        'position': args.position,
        'method':   'dora',
    }

    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = os.path.join(
        args.save_path, backbone, args.dataset,
        f'{args.shots}shots', f'seed{args.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{args.filename}.pt')
    torch.save({'weights': weights, 'metadata': metadata}, save_path)
    print(f'DoRA weights saved to {save_path}')


def load_dora(args, dora_layers):
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = os.path.join(
        args.save_path, backbone, args.dataset,
        f'{args.shots}shots', f'seed{args.seed}', f'{args.filename}.pt'
    )
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'DoRA weights not found at {load_path}')

    data = torch.load(load_path, map_location='cpu')
    meta = data['metadata']
    for key in ('r', 'alpha', 'encoder', 'params', 'position'):
        if meta[key] != getattr(args, key):
            raise ValueError(f'{key} mismatch: expected {getattr(args, key)}, got {meta[key]}')

    weights = data['weights']
    proj_map = [('q_proj', 'q_proj'), ('k_proj', 'k_proj'),
                ('v_proj', 'v_proj'), ('proj', 'proj')]
    for i, layer in enumerate(dora_layers):
        lw = weights[f'layer_{i}']
        for proj_key, proj_attr in proj_map:
            proj = getattr(layer, proj_attr)
            if isinstance(proj, LinearDoRA) and proj_key in lw:
                proj.lora_A.data.copy_(lw[proj_key]['lora_A'])
                proj.lora_B.data.copy_(lw[proj_key]['lora_B'])
                proj.magnitude.data.copy_(lw[proj_key]['magnitude'])

    print(f'DoRA weights loaded from {load_path}')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def clip_classifier(classnames, template, clip_model):
    templates = [template] if isinstance(template, str) else list(template)
    device = next(clip_model.parameters()).device

    texts = [t.format(name) for name in classnames for t in templates]
    tokenized = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokenized)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    C, T = len(classnames), len(templates)
    per_class = []
    for i in range(C):
        chunk = text_feats[i * T:(i + 1) * T]
        rep = chunk.mean(0)
        per_class.append(rep / rep.norm())

    return torch.stack(per_class, dim=0).t().to(device)  # (D, C)


def evaluate_dora(clip_model, loader, dataset):
    clip_model.eval()
    device = next(clip_model.parameters()).device

    with torch.no_grad():
        template = dataset.template[0]
        texts = clip.tokenize(
            [template.format(c.replace('_', ' ')) for c in dataset.classnames]
        ).to(device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            text_feats = clip_model.encode_text(texts)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    acc, total = 0., 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                img_feats = clip_model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = img_feats @ text_feats.t()
            acc += cls_acc(logits, target) * len(images)
            total += len(images)

    return acc / total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_dora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    device = next(clip_model.parameters()).device

    print('\nBuilding text classifier.')
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    print('Pre-loading test features.')
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features, test_labels = test_features.to(device), test_labels.to(device)

    # Zero-shot baseline
    zs_logits = logit_scale * test_features @ textual_features
    print(f'\n**** Zero-shot CLIP test accuracy: {cls_acc(zs_logits, test_labels):.2f} ****\n')

    test_features, test_labels = test_features.cpu(), test_labels.cpu()

    # Apply DoRA
    dora_layers = apply_dora(args, clip_model)
    clip_model = clip_model.to(device)
    print(f'Applied DoRA to {len(dora_layers)} attention modules.')

    if args.eval_only:
        load_dora(args, dora_layers)
        acc = evaluate_dora(clip_model, test_loader, dataset)
        print(f'**** Test accuracy: {acc:.2f} ****')
        return

    mark_only_dora_as_trainable(clip_model)
    n_trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}')

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(
        get_dora_parameters(clip_model),
        lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    count_iters = 0
    while count_iters < total_iters:
        clip_model.train()
        acc_train, loss_epoch, total_samples = 0., 0., 0

        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for images, target in tqdm(train_loader, desc=f'iter {count_iters}/{total_iters}'):
            template = dataset.template[0]
            texts = [template.format(c.replace('_', ' ')) for c in dataset.classnames]
            images, target = images.to(device), target.to(device)

            if args.encoder in ('text', 'both'):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    tok = clip.tokenize(texts).to(device)
                    class_emb = clip_model.encode_text(tok)
                text_features = class_emb / class_emb.norm(dim=-1, keepdim=True)

            if args.encoder in ('vision', 'both'):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feats = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        img_feats = clip_model.encode_image(images)

            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = logit_scale * img_feats @ text_features.t()
            loss = F.cross_entropy(logits, target)

            acc_train    += cls_acc(logits, target) * target.shape[0]
            loss_epoch   += loss.item() * target.shape[0]
            total_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters:
            lr = scheduler.get_last_lr()[0]
            print(f'LR: {lr:.6f}  Acc: {acc_train/total_samples:.4f}  Loss: {loss_epoch/total_samples:.4f}')

    acc_test = evaluate_dora(clip_model, test_loader, dataset)
    print(f'\n**** Final test accuracy: {acc_test:.2f} ****\n')

    if args.save_path is not None:
        save_dora(args, dora_layers)


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description='DoRA fine-tuning for CLIP')
    parser.add_argument('--seed',         default=1,        type=int)
    parser.add_argument('--root_path',    type=str,         default='')
    parser.add_argument('--dataset',      type=str,         default='eurosat')
    parser.add_argument('--shots',        default=16,       type=int)
    parser.add_argument('--backbone',     default='ViT-B/16', type=str)
    # Training
    parser.add_argument('--lr',           default=2e-4,     type=float)
    parser.add_argument('--n_iters',      default=100,      type=int)
    parser.add_argument('--batch_size',   default=32,       type=int)
    # DoRA / LoRA
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top1', 'top2', 'top3'])
    parser.add_argument('--encoder', type=str, default='both',
                        choices=['text', 'vision', 'both'])
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='which projections to apply DoRA to: q, k, v, o')
    parser.add_argument('--r',            default=4,        type=int,   help='LoRA rank')
    parser.add_argument('--alpha',        default=1.0,      type=float, help='LoRA alpha (scaling)')
    parser.add_argument('--dropout_rate', default=0.0,      type=float, help='dropout before DoRA projection')
    # Save / load
    parser.add_argument('--save_path',  default=None, help='directory to save DoRA weights')
    parser.add_argument('--filename',   default='dora_weights', help='filename stem for saved weights')
    parser.add_argument('--eval_only',  default=False, action='store_true')
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_arguments()
    set_random_seed(args.seed)

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    print('Preparing dataset.')
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # build_data_loader handles pin_memory internally; only num_workers is accepted
    dl_kwargs     = dict(num_workers=8, pin_memory=True)   # for direct DataLoader (imagenet)
    bdl_kwargs    = dict(num_workers=8)                    # for build_data_loader

    # Datasets that manage their own transforms return plain Dataset objects
    # (not Datum lists), so they must bypass build_data_loader's DatasetWrapper.
    raw_dataset = args.dataset in ('imagenet', 'medmnist')

    if raw_dataset:
        val_loader  = torch.utils.data.DataLoader(dataset.val,   batch_size=256, shuffle=False, **dl_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset.test,  batch_size=256, shuffle=False, **dl_kwargs)
        train_loader = (
            torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size,
                                        shuffle=True, **dl_kwargs)
            if not args.eval_only else None
        )
    else:
        val_loader  = build_data_loader(data_source=dataset.val,  batch_size=256, is_train=False,
                                        tfm=preprocess, shuffle=False, **bdl_kwargs)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False,
                                        tfm=preprocess, shuffle=False, **bdl_kwargs)
        train_loader = (
            build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size,
                              tfm=train_transform, is_train=True, shuffle=True, **bdl_kwargs)
            if not args.eval_only else None
        )

    run_dora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()
