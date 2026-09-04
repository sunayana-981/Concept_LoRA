"""
unified_finetune.py
====================
Fine-tune CLIP, SigLIP2, DINO, or ALIGN with either LoRA or DoRA on any of the
16 supported classification datasets.

Supported models : clip | siglip | dino | align
Supported methods: lora | dora

Example
-------
python unified_finetune.py \
    --model clip  --method lora  \
    --dataset eurosat  --root_path /data \
    --shots 16  --save_path ./weights

python unified_finetune.py \
    --model dino  --method dora \
    --dataset oxford_pets  --root_path /data \
    --shots 16  --save_path ./weights

python unified_finetune.py \
    --model align  --method lora \
    --dataset caltech101  --root_path /data/caltech-101 \
    --shots 16  --save_path ./weights
"""

import os
import sys
import csv
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

# ── project root on the path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CLIP
import CLIP_LoRA.clip as clip
from CLIP_LoRA.datasets import build_dataset, RAW_DATASETS
from CLIP_LoRA.datasets.utils import build_data_loader
from CLIP_LoRA.utils import cls_acc, pre_load_features

# existing CLIP-LoRA / DoRA helpers
from CLIP_LoRA.loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora,
)
from CLIP_LoRA.dora_finetune import (
    apply_dora,
    mark_only_dora_as_trainable,
    get_dora_parameters,
    save_dora,
    load_dora,
)

# ALIGN + DINOv2 + SigLIP2 via HuggingFace
from transformers import AlignModel, AutoTokenizer, Dinov2Model, AutoImageProcessor
try:
    # Despite the repository name, google/siglip2-base-patch16-224 currently
    # publishes a ``model_type: siglip`` config.  Loading it with Siglip2Model
    # selects the NaFlex input path (flattened patches + spatial shapes) and is
    # incompatible with the ordinary 4-D pixel tensors used by this trainer.
    from transformers import SiglipModel
except ImportError:
    SiglipModel = None


# =============================================================================
# Generic LoRA / DoRA Linear wrappers
# (used for DINO and ALIGN; CLIP keeps its own implementations)
# =============================================================================

class LinearLoRA(nn.Module):
    """LoRA adapter wrapping any nn.Linear layer."""

    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        d_out, d_in = existing_linear.weight.shape
        self.scaling = lora_alpha / r

        dev = existing_linear.weight.device
        self.weight = nn.Parameter(existing_linear.weight.data.clone(), requires_grad=False)
        if existing_linear.bias is not None:
            self.bias = nn.Parameter(existing_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        # lora_A: (r, d_in)  → F.linear(x, lora_A) = x @ lora_A.T → (B, r)
        # lora_B: (d_out, r) → F.linear(h, lora_B) = h @ lora_B.T → (B, d_out)
        lora_h = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora_h, self.lora_B)
        return base + self.scaling * lora


class LinearDoRA(nn.Module):
    """DoRA (weight-decomposed LoRA) adapter wrapping any nn.Linear layer."""

    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 4,
        lora_alpha: float = 1.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        d_out, d_in = existing_linear.weight.shape
        self.scaling = lora_alpha / math.sqrt(r)

        dev = existing_linear.weight.device
        self.weight = nn.Parameter(existing_linear.weight.data.clone(), requires_grad=False)
        if existing_linear.bias is not None:
            self.bias = nn.Parameter(existing_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        col_norms = existing_linear.weight.data.norm(dim=1)
        self.magnitude = nn.Parameter(col_norms.clone())

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        adapted = self.weight + delta_w
        col_norms = adapted.norm(dim=1, keepdim=True)
        weight_dora = (self.magnitude.unsqueeze(1) / col_norms) * adapted
        return F.linear(self.dropout(x), weight_dora, self.bias)


# Leaf parameter names that belong to LoRA / DoRA adapters
_LORA_NAMES = {'lora_A', 'lora_B'}
_DORA_NAMES = {'lora_A', 'lora_B', 'magnitude'}


def _make_wrapper(method: str, linear: nn.Linear, args) -> nn.Module:
    kwargs = dict(r=args.r, lora_alpha=args.alpha, dropout_rate=args.dropout_rate)
    return LinearDoRA(linear, **kwargs) if method == 'dora' else LinearLoRA(linear, **kwargs)


def _freeze_all_unfreeze_adapters(model: nn.Module, method: str) -> None:
    trainable = _DORA_NAMES if method == 'dora' else _LORA_NAMES
    for name, param in model.named_parameters():
        param.requires_grad = name.split('.')[-1] in trainable


def _get_adapter_params(model: nn.Module, method: str):
    trainable = _DORA_NAMES if method == 'dora' else _LORA_NAMES
    return [p for n, p in model.named_parameters() if n.split('.')[-1] in trainable]


def _classification_loss(logits, targets, multilabel=False):
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    return F.cross_entropy(logits, targets)


def _classification_acc(logits, targets, multilabel=False):
    if multilabel:
        predictions = logits.float() > 0
        return (predictions == targets.bool()).float().mean().item() * 100.0
    return cls_acc(logits, targets)


def _capture_trainable_state(*modules):
    return [
        {
            name: param.detach().cpu().clone()
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        for module in modules
    ]


def _restore_trainable_state(states, *modules):
    with torch.no_grad():
        for state, module in zip(states, modules):
            for name, param in module.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param.device))


# =============================================================================
# Layer position index maps (shared across models)
# =============================================================================

_POS_12 = {
    'top1':        [11],
    'top3':        [9, 10, 11],
    'bottom':      [0, 1, 2, 3],
    'mid':         [4, 5, 6, 7],
    'up':          [8, 9, 10, 11],
    'half-up':     [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all':         list(range(12)),
}


# =============================================================================
# Results CSV logging
# =============================================================================

_CSV_FIELDS = [
    'model', 'method', 'dataset', 'shots', 'seed',
    'backbone', 'encoder', 'position', 'r', 'alpha',
    'baseline_type',
    'zs_acc', 'lp_acc', 'ft_acc', 'adapter_delta',
]


def _results_row_key(row):
    return (
        row['model'], row['method'], row['dataset'], row['shots'], row['seed'],
        row['backbone'], row['encoder'], row['position'], row['r'], row['alpha'],
        row['baseline_type'],
    )


def _append_results_csv(args, zs_acc, ft_acc, lp_acc=None):
    """Upsert one row in the results CSV and drop duplicate keys."""
    if not args.results_csv:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    if args.model in ('clip', 'siglip'):
        encoder = args.encoder
    elif args.model == 'align':
        encoder = 'text'
    else:
        encoder = 'vision'

    baseline_type = 'prototype_kshot+linear_probe' if args.model == 'dino' else 'zero_shot'
    adapter_delta = (ft_acc - lp_acc) if (lp_acc is not None) else (ft_acc - zs_acc)

    new_row = {
        'model':    args.model,
        'method':   args.method,
        'dataset':  args.dataset,
        'shots':    str(args.shots),
        'seed':     str(args.seed),
        'backbone': (
            args.backbone if args.model == 'clip' else
            DINO_MODEL_ID if args.model == 'dino' else
            SIGLIP_MODEL_ID if args.model == 'siglip' else
            ALIGN_MODEL_ID
        ),
        'encoder':  encoder,
        'position': args.position,
        'r':        str(args.r),
        'alpha':    str(args.alpha),
        'baseline_type': baseline_type,
        'zs_acc':   f'{zs_acc:.4f}',
        'lp_acc':   '' if lp_acc is None else f'{lp_acc:.4f}',
        'ft_acc':   f'{ft_acc:.4f}',
        'adapter_delta': f'{adapter_delta:.4f}',
    }
    new_key = _results_row_key(new_row)

    rows = []
    if os.path.exists(args.results_csv):
        with open(args.results_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {k: row.get(k, '') for k in _CSV_FIELDS}
                rows.append(normalized)

    deduped = []
    updated = False
    seen = set()
    for row in rows:
        key = _results_row_key(row)
        if key == new_key:
            if not updated:
                deduped.append(new_row)
                updated = True
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if not updated:
        deduped.append(new_row)

    with open(args.results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(deduped)
    if updated:
        print(f'Results updated in {args.results_csv}')
    else:
        print(f'Results appended to {args.results_csv}')


# =============================================================================
# DINO nearest-centroid k-shot prototype baseline
# (vision-only model has no text encoder)
# =============================================================================

def _dino_forward(model, images):
    """Extract CLS-token features from DINOv2 (HuggingFace API)."""
    return model(pixel_values=images).pooler_output


def _dino_prototype_acc(
    model, train_loader, test_loader, num_classes, device, multilabel=False
):
    """Classify test images by cosine similarity to per-class k-shot centroids."""
    model.eval()

    # --- build prototypes from k-shot train set ---
    all_feats, all_labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(train_loader, desc='DINO prototype: train feats', leave=False):
            images = images.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                feats = _dino_forward(model, images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.float().cpu())
            all_labels.append(targets.cpu())

    all_feats  = torch.cat(all_feats)   # (N_train, D)
    all_labels = torch.cat(all_labels)

    prototypes = torch.zeros(num_classes, all_feats.size(1))
    for c in range(num_classes):
        mask = all_labels[:, c] > 0 if multilabel else all_labels == c
        if mask.sum() > 0:
            proto = all_feats[mask].mean(0)
            prototypes[c] = proto / (proto.norm() + 1e-8)
    prototypes = prototypes.to(device)

    # --- nearest centroid on test set ---
    acc, total = 0., 0
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='DINO prototype: test eval', leave=False):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                feats = _dino_forward(model, images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats.float() @ prototypes.t()
            acc += _classification_acc(logits, targets, multilabel) * len(images)
            total += len(images)

    return acc / total


def _evaluate_dino(model, head, loader, device, desc='DINO eval', multilabel=False):
    model.eval()
    head.eval()
    acc, total = 0., 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                feats = _dino_forward(model, images)
                logits = head(feats.float())
            acc += _classification_acc(logits, targets, multilabel) * len(images)
            total += len(images)
    return acc / total


# =============================================================================
# CLIP helpers (reuse existing implementations)
# =============================================================================

def _clip_classifier(classnames, template, model, device):
    if isinstance(template, (list, tuple)):
        templates = list(template)
    else:
        templates = [template]
    texts = [t.format(name.replace('_', ' ')) for name in classnames for t in templates]
    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokenized)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    C, T = len(classnames), len(templates)
    per_class = []
    for i in range(C):
        chunk = feats[i * T:(i + 1) * T]
        rep = chunk.mean(0)
        per_class.append(rep / rep.norm())
    return torch.stack(per_class, dim=0).t().to(device)  # (D, C)


def _evaluate_clip(model, loader, dataset, device):
    model.eval()
    template = dataset.template[0]
    texts = clip.tokenize(
        [template.format(c.replace('_', ' ')) for c in dataset.classnames]
    ).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            text_feats = model.encode_text(texts)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    acc, total = 0., 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            acc += _classification_acc(
                img_feats @ text_feats.t(), target, getattr(dataset, 'multilabel', False)
            ) * len(images)
            total += len(images)
    return acc / total


def run_clip(args, dataset, train_loader, val_loader, test_loader):
    device = 'cuda'
    clip_model, _ = clip.load(args.backbone)
    clip_model = clip_model.float().to(device)
    clip_model.eval()
    logit_scale = 100

    textual_features = _clip_classifier(
        dataset.classnames, dataset.template, clip_model, device
    )
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    test_features, test_labels = test_features.to(device), test_labels.to(device)

    zs_logits = logit_scale * test_features @ textual_features
    multilabel = getattr(dataset, 'multilabel', False)
    zs_acc = _classification_acc(zs_logits, test_labels, multilabel)
    print(f'\n**** Zero-shot CLIP test accuracy: {zs_acc:.2f} ****\n')
    test_features, test_labels = test_features.cpu(), test_labels.cpu()

    if args.method == 'lora':
        list_layers = apply_lora(args, clip_model)
        clip_model = clip_model.to(device)
        mark_only_lora_as_trainable(clip_model)
        trainable_params = get_lora_parameters(clip_model)
    else:
        list_layers = apply_dora(args, clip_model)
        clip_model = clip_model.to(device)
        mark_only_dora_as_trainable(clip_model)
        trainable_params = get_dora_parameters(clip_model)

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2,
                                   betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    best_val_acc = _evaluate_clip(clip_model, val_loader, dataset, device)
    best_state = _capture_trainable_state(clip_model)

    count_iters = 0
    while count_iters < total_iters:
        clip_model.train()
        if args.encoder == 'vision':
            text_features = textual_features.t().half()

        for images, target in tqdm(train_loader, desc=f'CLIP/{args.method} iter {count_iters}'):
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
            loss = _classification_loss(logits, target, multilabel)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters:
            lr = scheduler.get_last_lr()[0]
            print(f'LR: {lr:.6f}')

        val_acc = _evaluate_clip(clip_model, val_loader, dataset, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = _capture_trainable_state(clip_model)
            print(f'CLIP best val accuracy updated: {best_val_acc:.2f}')

    _restore_trainable_state(best_state, clip_model)

    ft_acc = _evaluate_clip(clip_model, test_loader, dataset, device)
    print(f'\n**** CLIP/{args.method} final test accuracy: {ft_acc:.2f} ****\n')

    if args.save_path is not None:
        if args.method == 'lora':
            save_lora(args, list_layers)
        else:
            save_dora(args, list_layers)

    return zs_acc, ft_acc


# =============================================================================
# DINO (facebook/dinov2-base via HuggingFace Transformers)
# =============================================================================

DINO_MODEL_ID = 'facebook/dinov2-base'


def _get_dino_preprocess(training: bool):
    # DINOv2: resize shortest edge to 256, center-crop 224, ImageNet normalisation
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


def _apply_lora_dora_dino(model: Dinov2Model, args) -> list:
    """Apply LoRA/DoRA to DINOv2's transformer encoder attention layers.

    DINOv2 (HuggingFace) has separate query/key/value projections in
    encoder.layer[i].attention.attention.{query,key,value} and an output
    projection at encoder.layer[i].attention.output.dense.
    """
    indices = _POS_12[args.position]
    adapted = []
    for i, layer in enumerate(model.encoder.layer):
        if i not in indices:
            continue
        attn_self = layer.attention.attention
        if 'q' in args.params:
            attn_self.query = _make_wrapper(args.method, attn_self.query, args)
            adapted.append(attn_self.query)
        if 'k' in args.params:
            attn_self.key = _make_wrapper(args.method, attn_self.key, args)
            adapted.append(attn_self.key)
        if 'v' in args.params:
            attn_self.value = _make_wrapper(args.method, attn_self.value, args)
            adapted.append(attn_self.value)
        if 'o' in args.params:
            out_dense = layer.attention.output.dense
            layer.attention.output.dense = _make_wrapper(args.method, out_dense, args)
            adapted.append(layer.attention.output.dense)
    return adapted


def _save_dino(args, model: nn.Module, head: nn.Linear, adapted_layers: list):
    backbone = 'dinov2-base'
    save_dir = os.path.join(
        args.save_path, f'dino_{backbone}', args.dataset,
        f'{args.shots}shots', f'seed{args.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    trainable = _DORA_NAMES if args.method == 'dora' else _LORA_NAMES
    adapter_state = {
        n: p.data.cpu()
        for n, p in model.named_parameters()
        if n.split('.')[-1] in trainable
    }
    torch.save({
        'adapter': adapter_state,
        'head': head.state_dict(),
        'metadata': {
            'method': args.method, 'r': args.r, 'alpha': args.alpha,
            'position': args.position, 'params': args.params,
        },
    }, os.path.join(save_dir, f'{args.filename}.pt'))
    print(f'DINO weights saved to {save_dir}/{args.filename}.pt')


def run_dino(args, dataset, train_loader, val_loader, test_loader):
    device = 'cuda'
    num_classes = len(dataset.classnames)
    multilabel = getattr(dataset, 'multilabel', False)

    print('Loading DINOv2 (facebook/dinov2-base)…')
    model = Dinov2Model.from_pretrained(DINO_MODEL_ID, local_files_only=True)
    model = model.to(device)
    model.eval()

    feat_dim = 768  # DINOv2 ViT-B/14 CLS token dimension
    total_iters = args.n_iters * args.shots

    # Baseline 1: nearest-centroid on frozen DINO features (using k-shot train split)
    print('Computing DINO prototype baseline (k-shot nearest centroid)…')
    zs_acc = _dino_prototype_acc(
        model, train_loader, test_loader, num_classes, device, multilabel
    )
    print(f'\n**** DINO prototype (k-shot nearest centroid) accuracy: {zs_acc:.2f} ****\n')

    # Baseline 2: linear probe on frozen DINO features (skippable via --no_linear_probe)
    lp_acc = None
    if not args.no_linear_probe:
        print('Training DINO linear probe baseline (frozen backbone)…')
        head_lp = nn.Linear(feat_dim, num_classes, bias=True).to(device)
        nn.init.trunc_normal_(head_lp.weight, std=0.02)
        nn.init.zeros_(head_lp.bias)

        lp_params = list(head_lp.parameters())
        optimizer_lp = torch.optim.AdamW(
            lp_params, lr=args.lp_lr, weight_decay=1e-2, betas=(0.9, 0.999)
        )
        scheduler_lp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lp, total_iters, eta_min=1e-6)
        scaler_lp = torch.cuda.amp.GradScaler()

        count_iters = 0
        while count_iters < total_iters:
            model.eval()
            head_lp.train()
            acc_sum, loss_sum, n = 0., 0., 0

            for images, targets in tqdm(train_loader, desc=f'DINO/linear-probe iter {count_iters}'):
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        feats = _dino_forward(model, images)
                    logits = head_lp(feats.float())
                    loss = _classification_loss(logits, targets, multilabel)

                acc_sum += _classification_acc(logits, targets, multilabel) * len(images)
                loss_sum += loss.item() * len(images)
                n += len(images)

                optimizer_lp.zero_grad()
                scaler_lp.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler_lp.unscale_(optimizer_lp)
                    torch.nn.utils.clip_grad_norm_(lp_params, args.max_grad_norm)
                scaler_lp.step(optimizer_lp)
                scaler_lp.update()
                scheduler_lp.step()
                count_iters += 1
                if count_iters >= total_iters:
                    break

            if count_iters < total_iters and n > 0:
                lr = scheduler_lp.get_last_lr()[0]
                print(f'LP LR: {lr:.6f}  Acc: {acc_sum/n:.4f}  Loss: {loss_sum/n:.4f}')

        lp_acc = _evaluate_dino(
            model, head_lp, test_loader, device,
            desc='DINO linear probe eval', multilabel=multilabel,
        )
        print(f'\n**** DINO linear probe (frozen backbone) accuracy: {lp_acc:.2f} ****\n')

    # LoRA/DoRA + trainable head
    head = nn.Linear(feat_dim, num_classes, bias=True).to(device)
    nn.init.trunc_normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)

    adapted_layers = _apply_lora_dora_dino(model, args)
    _freeze_all_unfreeze_adapters(model, args.method)
    head.requires_grad_(True)

    adapter_params = _get_adapter_params(model, args.method)
    head_params = list(head.parameters())
    trainable_params = adapter_params + head_params
    n_params = sum(p.numel() for p in trainable_params)
    print(f'DINO trainable params: {n_params:,}')

    optimizer = torch.optim.AdamW(
        [
            {'params': adapter_params, 'lr': args.lr},
            {'params': head_params, 'lr': args.lr * args.head_lr_mult},
        ],
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    best_val_acc = _evaluate_dino(
        model, head, val_loader, device, desc='DINO initial val', multilabel=multilabel
    )
    best_state = _capture_trainable_state(model, head)

    count_iters = 0
    while count_iters < total_iters:
        model.train()
        head.train()
        acc_sum, loss_sum, n = 0., 0., 0

        for images, targets in tqdm(train_loader, desc=f'DINO/{args.method} iter {count_iters}'):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                feats = _dino_forward(model, images)
                logits = head(feats.float())
                loss = _classification_loss(logits, targets, multilabel)

            acc_sum += _classification_acc(logits, targets, multilabel) * len(images)
            loss_sum += loss.item() * len(images)
            n += len(images)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters and n > 0:
            lr = scheduler.get_last_lr()[0]
            print(f'LR: {lr:.6f}  Acc: {acc_sum/n:.4f}  Loss: {loss_sum/n:.4f}')

        val_acc = _evaluate_dino(
            model, head, val_loader, device, desc='DINO val', multilabel=multilabel
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = _capture_trainable_state(model, head)
            print(f'DINO best val accuracy updated: {best_val_acc:.2f}')

    _restore_trainable_state(best_state, model, head)

    ft_acc = _evaluate_dino(
        model, head, test_loader, device, desc='DINO eval', multilabel=multilabel
    )
    print(f'\n**** DINO/{args.method} final test accuracy: {ft_acc:.2f} ****\n')
    if lp_acc is not None:
        print(f'**** DINO adaptation effect (LoRA+head - linear probe): {ft_acc - lp_acc:+.2f} ****\n')

    if args.save_path is not None:
        _save_dino(args, model, head, adapted_layers)

    return zs_acc, lp_acc, ft_acc


# =============================================================================
# SigLIP2 (google/siglip2-base-patch16-224 via HuggingFace Transformers)
# =============================================================================

SIGLIP_MODEL_ID = 'google/siglip2-base-patch16-224'


def _get_siglip_preprocess(training: bool):
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _apply_lora_dora_siglip(model: nn.Module, args) -> list:
    """Apply adapters to selected SigLIP2 text/vision attention projections."""
    indices = _POS_12[args.position]
    adapted = []
    branches = []
    if args.encoder in ('text', 'both'):
        branches.append(model.text_model.encoder.layers)
    if args.encoder in ('vision', 'both'):
        branches.append(model.vision_model.encoder.layers)

    projection_names = {
        'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'o': 'out_proj',
    }
    for layers in branches:
        for i, layer in enumerate(layers):
            if i not in indices:
                continue
            attn = layer.self_attn
            for requested, attr in projection_names.items():
                if requested not in args.params:
                    continue
                wrapped = _make_wrapper(args.method, getattr(attn, attr), args)
                setattr(attn, attr, wrapped)
                adapted.append(wrapped)
    return adapted


def _siglip_text_features(model, tokenizer, classnames, template, device, grad=False):
    tmpl = template[0] if isinstance(template, (list, tuple)) else template
    texts = [tmpl.format(c.replace('_', ' ')) for c in classnames]
    tokens = tokenizer(
        texts, return_tensors='pt', padding='max_length', truncation=True,
        max_length=64,
    ).to(device)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            feats = model.get_text_features(**tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def _siglip_logits(model, image_features, text_features):
    scale = model.logit_scale.exp().clamp(max=100).float()
    bias = model.logit_bias.float() if getattr(model, 'logit_bias', None) is not None else 0.0
    return scale * image_features.float() @ text_features.t().float() + bias


def _evaluate_siglip(model, tokenizer, loader, dataset, device, desc='SigLIP2 eval'):
    model.eval()
    text_features = _siglip_text_features(
        model, tokenizer, dataset.classnames, dataset.template, device, grad=False
    )
    acc, total = 0.0, 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = _siglip_logits(model, image_features, text_features)
            acc += _classification_acc(
                logits, targets, getattr(dataset, 'multilabel', False)
            ) * len(images)
            total += len(images)
    return acc / total


def _save_siglip(args, model: nn.Module):
    save_dir = os.path.join(
        args.save_path, 'siglip2-base-patch16-224', args.dataset,
        f'{args.shots}shots', f'seed{args.seed}',
    )
    os.makedirs(save_dir, exist_ok=True)
    trainable = _DORA_NAMES if args.method == 'dora' else _LORA_NAMES
    adapter_state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if n.split('.')[-1] in trainable
    }
    checkpoint = os.path.join(save_dir, f'{args.filename}.pt')
    torch.save({
        'adapter': adapter_state,
        'metadata': {
            'model_id': SIGLIP_MODEL_ID,
            'method': args.method,
            'r': args.r,
            'alpha': args.alpha,
            'position': args.position,
            'params': args.params,
            'encoder': args.encoder,
        },
    }, checkpoint)
    print(f'SigLIP2 weights saved to {checkpoint}')


def run_siglip(args, dataset, train_loader, val_loader, test_loader):
    if SiglipModel is None:
        raise RuntimeError(
            'SigLIP requires a newer Transformers build. Run this model with '
            '/home/sunayana/miniconda3/envs/dncbm310/bin/python.'
        )

    device = 'cuda'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    print(f'Loading SigLIP2 ({SIGLIP_MODEL_ID})...')
    model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True)
    model.eval()
    multilabel = getattr(dataset, 'multilabel', False)

    zs_acc = _evaluate_siglip(
        model, tokenizer, test_loader, dataset, device, desc='SigLIP2 zero-shot eval'
    )
    print(f'\n**** SigLIP2 zero-shot test accuracy: {zs_acc:.2f} ****\n')

    adapted = _apply_lora_dora_siglip(model, args)
    if not adapted:
        raise ValueError('No SigLIP2 layers were adapted; check --encoder/--params/--position')
    _freeze_all_unfreeze_adapters(model, args.method)
    trainable_params = _get_adapter_params(model, args.method)
    print(
        f'SigLIP2 adapted projections: {len(adapted)}; '
        f'trainable params: {sum(p.numel() for p in trainable_params):,}'
    )

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(total_iters, 1), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()
    best_val_acc = _evaluate_siglip(
        model, tokenizer, val_loader, dataset, device, desc='SigLIP initial val'
    )
    best_state = _capture_trainable_state(model)

    count_iters = 0
    while count_iters < total_iters:
        model.train()
        for images, targets in tqdm(train_loader, desc=f'SigLIP2/{args.method} iter {count_iters}'):
            images, targets = images.to(device), targets.to(device)
            text_grad = args.encoder in ('text', 'both')
            vision_grad = args.encoder in ('vision', 'both')
            text_features = _siglip_text_features(
                model, tokenizer, dataset.classnames, dataset.template, device,
                grad=text_grad,
            )
            vision_context = torch.enable_grad() if vision_grad else torch.no_grad()
            with vision_context:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    image_features = model.get_image_features(pixel_values=images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = _siglip_logits(model, image_features, text_features)
            loss = _classification_loss(logits, targets, multilabel)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters:
            val_acc = _evaluate_siglip(
                model, tokenizer, val_loader, dataset, device, desc='SigLIP2 val'
            )
            print(f'LR: {scheduler.get_last_lr()[0]:.6f}  Val acc: {val_acc:.2f}')
        else:
            val_acc = _evaluate_siglip(
                model, tokenizer, val_loader, dataset, device, desc='SigLIP2 final val'
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = _capture_trainable_state(model)
            print(f'SigLIP best val accuracy updated: {best_val_acc:.2f}')

    _restore_trainable_state(best_state, model)

    ft_acc = _evaluate_siglip(model, tokenizer, test_loader, dataset, device)
    print(f'\n**** SigLIP2/{args.method} final test accuracy: {ft_acc:.2f} ****\n')
    if args.save_path is not None:
        _save_siglip(args, model)
    return zs_acc, ft_acc


# =============================================================================
# ALIGN (HuggingFace kakaobrain/align-base)
# =============================================================================

ALIGN_MODEL_ID = 'kakaobrain/align-base'


def _get_align_logit_scale(model: AlignModel) -> float:
    if hasattr(model, 'temperature'):
        temp = model.temperature.detach().float().item()
        if temp > 0:
            return 1.0 / temp
    return 100.0


def _get_align_preprocess(training: bool, image_size: int = 289):
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _apply_lora_dora_align(model: AlignModel, args) -> list:
    """Apply LoRA/DoRA to ALIGN text-encoder attention layers."""
    indices = _POS_12[args.position]
    adapted = []
    for i, layer in enumerate(model.text_model.encoder.layer):
        if i not in indices:
            continue
        attn_self = layer.attention.self
        if 'q' in args.params:
            attn_self.query = _make_wrapper(args.method, attn_self.query, args)
            adapted.append(attn_self.query)
        if 'k' in args.params:
            attn_self.key = _make_wrapper(args.method, attn_self.key, args)
            adapted.append(attn_self.key)
        if 'v' in args.params:
            attn_self.value = _make_wrapper(args.method, attn_self.value, args)
            adapted.append(attn_self.value)
        if 'o' in args.params:
            out_dense = layer.attention.output.dense
            layer.attention.output.dense = _make_wrapper(args.method, out_dense, args)
            adapted.append(layer.attention.output.dense)
    return adapted


def _set_align_trainability(model: AlignModel, args):
    """Configure trainable params for ALIGN based on --encoder.

    ALIGN's vision tower is EfficientNet rather than a transformer, so the
    q/k/v LoRA/DoRA adapters in this runner apply to the text transformer only.
    The full vision tower must remain frozen; training it at the adapter LR
    corrupts BatchNorm statistics and causes catastrophic accuracy collapse.
    """
    train_text = True
    train_vision = False
    trainable_leaves = _DORA_NAMES if args.method == 'dora' else _LORA_NAMES

    for p in model.parameters():
        p.requires_grad = False

    if train_text:
        for name, p in model.named_parameters():
            if name.startswith('text_model.') and name.split('.')[-1] in trainable_leaves:
                p.requires_grad = True

    trainable_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    return train_text, train_vision, trainable_param_names


def _build_align_text_features(model, tokenizer, classnames, template, device):
    tmpl = template[0] if isinstance(template, (list, tuple)) else template
    texts = [tmpl.format(c.replace('_', ' ')) for c in classnames]
    with torch.no_grad():
        inputs = tokenizer(
            texts, return_tensors='pt', padding=True,
            truncation=True, max_length=64,
        ).to(device)
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.t()  # (D, C)


def _save_align(args, model: AlignModel):
    backbone = 'align-base'
    save_dir = os.path.join(
        args.save_path, backbone, args.dataset,
        f'{args.shots}shots', f'seed{args.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    trainable = _DORA_NAMES if args.method == 'dora' else _LORA_NAMES
    adapter_state = {
        n: p.data.cpu()
        for n, p in model.named_parameters()
        if n.split('.')[-1] in trainable
    }
    # For encoder=vision/both, include all trainable weights so ALIGN checkpoints
    # preserve vision updates in addition to text adapters.
    trainable_state = {
        n: p.data.cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    torch.save({
        'adapter': adapter_state,
        'trainable': trainable_state,
        'metadata': {
            'method': args.method, 'r': args.r, 'alpha': args.alpha,
            'position': args.position, 'params': args.params,
            'encoder': args.encoder,
        },
    }, os.path.join(save_dir, f'{args.filename}.pt'))
    print(f'ALIGN weights saved to {save_dir}/{args.filename}.pt')


def run_align(args, dataset, train_loader, _val_loader, test_loader):
    device = 'cuda'
    multilabel = getattr(dataset, 'multilabel', False)

    if args.encoder != 'text':
        print(
            f'ALIGN requested --encoder {args.encoder!r}; using text-only adapters. '
            'The EfficientNet vision tower is frozen to preserve BatchNorm statistics.'
        )
        args.encoder = 'text'

    print('Loading ALIGN model…')
    # Avoid tokenizer/fork deadlocks with DataLoader workers.
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    model = AlignModel.from_pretrained(ALIGN_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ALIGN_MODEL_ID)
    model = model.to(device)
    model.eval()
    logit_scale = _get_align_logit_scale(model)
    print(f'ALIGN logit scale set from model temperature: {logit_scale:.4f}')

    _apply_lora_dora_align(model, args)
    train_text, train_vision, _trainable_names = _set_align_trainability(model, args)
    if train_text:
        print('ALIGN adaptation mode: text adapters only (vision encoder frozen).')
    else:
        raise ValueError('ALIGN has no trainable params: set --encoder to text, vision, or both')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError('No trainable parameters selected for ALIGN')
    n_params = sum(p.numel() for p in trainable_params)
    print(f'ALIGN trainable params: {n_params:,}')

    def _align_acc(curr_model, loader, txt_feats):
        if loader is None:
            return None
        acc, total = 0.0, 0
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feats = curr_model.get_image_features(pixel_values=images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                logits = logit_scale * img_feats.float() @ txt_feats.float()
                acc += _classification_acc(logits, targets, multilabel) * len(images)
                total += len(images)
        return acc / total if total > 0 else None

    # Zero-shot baseline using the full (un-adapted) text encoder
    textual_features = _build_align_text_features(
        model, tokenizer, dataset.classnames, dataset.template, device
    )
    print('Computing zero-shot ALIGN baseline…')
    acc_zs, total = 0., 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                img_feats = model.get_image_features(pixel_values=images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = logit_scale * img_feats.float() @ textual_features.float()
            acc_zs += _classification_acc(logits, targets, multilabel) * len(images)
            total += len(images)
    zs_acc = acc_zs / total
    print(f'\n**** Zero-shot ALIGN test accuracy: {zs_acc:.2f} ****\n')

    # Track best adapter checkpoint on val to avoid catastrophic late-epoch drift.
    model.eval()
    val_txt = _build_align_text_features(
        model, tokenizer, dataset.classnames, dataset.template, device
    )
    best_val_acc = _align_acc(model, _val_loader, val_txt)
    best_adapter_state = {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2,
                                   betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    count_iters = 0
    while count_iters < total_iters:
        model.train()
        # ALIGN uses EfficientNet. Keep its BatchNorm buffers immutable.
        model.vision_model.eval()
        if not train_text:
            model.text_model.eval()
        tmpl = dataset.template[0]

        for images, targets in tqdm(train_loader, desc=f'ALIGN/{args.method} iter {count_iters}'):
            texts = [tmpl.format(c.replace('_', ' ')) for c in dataset.classnames]
            images, targets = images.to(device), targets.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Image branch: train only when vision encoder is requested.
                if train_vision:
                    img_feats = model.get_image_features(pixel_values=images)
                else:
                    with torch.no_grad():
                        img_feats = model.get_image_features(pixel_values=images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                if train_text:
                    tok = tokenizer(
                        texts, return_tensors='pt', padding=True,
                        truncation=True, max_length=64,
                    ).to(device)
                    text_feats = model.get_text_features(**tok)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                else:
                    with torch.no_grad():
                        tok = tokenizer(
                            texts, return_tensors='pt', padding=True,
                            truncation=True, max_length=64,
                        ).to(device)
                        text_feats = model.get_text_features(**tok)
                        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            logits = logit_scale * img_feats.float() @ text_feats.t().float()
            loss = _classification_loss(logits, targets, multilabel)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count_iters += 1
            if count_iters >= total_iters:
                break

        if count_iters < total_iters:
            lr = scheduler.get_last_lr()[0]
            print(f'LR: {lr:.6f}')

        # End-of-epoch val check; keep best adapter weights.
        model.eval()
        txt_feats = _build_align_text_features(
            model, tokenizer, dataset.classnames, dataset.template, device
        )
        val_acc = _align_acc(model, _val_loader, txt_feats)
        if val_acc is not None and (best_val_acc is None or val_acc > best_val_acc):
            best_val_acc = val_acc
            best_adapter_state = {
                n: p.detach().cpu().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            print(f'ALIGN best val accuracy updated: {best_val_acc:.2f}')

    # Restore best-on-val adapter params before final test evaluation.
    if best_adapter_state:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_adapter_state:
                    p.copy_(best_adapter_state[n].to(p.device))

    # Final evaluation
    model.eval()
    textual_features = _build_align_text_features(
        model, tokenizer, dataset.classnames, dataset.template, device
    )
    acc, total = 0., 0
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='ALIGN eval'):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                img_feats = model.get_image_features(pixel_values=images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = logit_scale * img_feats.float() @ textual_features.float()
            acc += _classification_acc(logits, targets, multilabel) * len(images)
            total += len(images)

    ft_acc = acc / total
    print(f'\n**** ALIGN/{args.method} final test accuracy: {ft_acc:.2f} ****\n')

    if args.save_path is not None:
        _save_align(args, model)

    return zs_acc, ft_acc


# =============================================================================
# DataLoader builders (model-aware preprocessing)
# =============================================================================

def _set_transform(split, tfm):
    if split is None:
        return
    if hasattr(split, 'transform'):
        split.transform = tfm
    elif hasattr(split, 'dataset') and hasattr(split.dataset, 'transform'):
        split.dataset.transform = tfm


def _make_loaders(args, dataset, clip_preprocess=None):
    """Return (train_loader, val_loader, test_loader) using model-appropriate preprocessing."""
    model = args.model
    is_raw = args.dataset in RAW_DATASETS  # medmnist / chexpert have baked transforms

    # Choose preprocessing transforms per model
    if model == 'clip':
        test_tfm = clip_preprocess
        train_tfm = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                  std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    elif model == 'dino':
        test_tfm = _get_dino_preprocess(training=False)
        train_tfm = _get_dino_preprocess(training=True)
    elif model == 'siglip':
        test_tfm = _get_siglip_preprocess(training=False)
        train_tfm = _get_siglip_preprocess(training=True)
    elif model == 'align':
        test_tfm = _get_align_preprocess(training=False)
        train_tfm = _get_align_preprocess(training=True)

    # HuggingFace tokenizers can deadlock with forked DataLoader workers.
    workers = 0 if args.model in ('align', 'siglip') else 8
    dl_kw = dict(num_workers=workers, pin_memory=True)
    # Keep evaluation below the 24 GB card's safe limit while another user may
    # share the GPU. This also avoids the former 2.31 GiB allocation spikes.
    if args.model == 'siglip':
        eval_batch_size = 128
    elif args.model == 'align':
        eval_batch_size = 64
    elif args.model == 'dino':
        eval_batch_size = 128
    else:
        eval_batch_size = 256

    if is_raw:
        # Raw torch Dataset objects (ImageFolder/medmnist/chexpert):
        # set model-specific transforms directly on the split datasets.
        _set_transform(dataset.train_x, train_tfm)
        _set_transform(dataset.val, test_tfm)
        _set_transform(dataset.test, test_tfm)

        test_loader = torch.utils.data.DataLoader(
            dataset.test, batch_size=eval_batch_size, shuffle=False, **dl_kw
        )
        val_loader = torch.utils.data.DataLoader(
            dataset.val, batch_size=eval_batch_size, shuffle=False, **dl_kw
        )
        train_loader = None
        if not args.eval_only:
            train_loader = torch.utils.data.DataLoader(
                dataset.train_x, batch_size=args.batch_size, shuffle=True, **dl_kw
            )
    else:
        bdl_kw = dict(num_workers=workers)
        val_loader = build_data_loader(
            data_source=dataset.val, batch_size=eval_batch_size, is_train=False,
            tfm=test_tfm, shuffle=False, **bdl_kw
        )
        test_loader = build_data_loader(
            data_source=dataset.test, batch_size=eval_batch_size, is_train=False,
            tfm=test_tfm, shuffle=False, **bdl_kw
        )
        train_loader = None
        if not args.eval_only:
            train_loader = build_data_loader(
                data_source=dataset.train_x, batch_size=args.batch_size,
                tfm=train_tfm, is_train=True, shuffle=True, **bdl_kw
            )

    return train_loader, val_loader, test_loader


# =============================================================================
# Argument parsing & entry point
# =============================================================================

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Unified LoRA/DoRA fine-tuning for CLIP, SigLIP2, DINO, and ALIGN'
    )
    # Core
    parser.add_argument('--model',   required=True, choices=['clip', 'siglip', 'dino', 'align'])
    parser.add_argument('--method',  required=True, choices=['lora', 'dora'])
    parser.add_argument('--seed',    default=1, type=int)

    # Dataset
    parser.add_argument('--dataset',    required=True, type=str)
    parser.add_argument('--root_path',  required=True, type=str)
    parser.add_argument('--shots',      default=16, type=int)

    # CLIP backbone (ignored for siglip/dino/align)
    parser.add_argument('--backbone', default='ViT-B/16', type=str)

    # Training
    parser.add_argument('--lr',         default=2e-4, type=float)
    parser.add_argument('--lp_lr',      default=1e-3, type=float,
                        help='DINO linear-probe head LR (frozen backbone baseline)')
    parser.add_argument('--head_lr_mult', default=10.0, type=float,
                        help='DINO head LR multiplier relative to --lr')
    parser.add_argument('--n_iters',    default=100,  type=int)
    parser.add_argument('--batch_size', default=32,   type=int)
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Gradient clipping norm (<=0 disables clipping)')

    # LoRA/DoRA hyperparameters
    parser.add_argument('--position', default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom',
                                 'all', 'top1', 'top3'])
    parser.add_argument('--encoder', default='both', choices=['text', 'vision', 'both'],
                        help='Which encoder to adapt (CLIP/SigLIP2; ALIGN also accepts it)')
    parser.add_argument('--params', metavar='N', nargs='+', default=['q', 'k', 'v'],
                        help='Attention projections to adapt: q k v o')
    parser.add_argument('--r',            default=4,   type=int,   help='LoRA rank')
    parser.add_argument('--alpha',        default=1.0, type=float, help='LoRA alpha')
    parser.add_argument('--dropout_rate', default=0.0, type=float)

    # Save / eval
    parser.add_argument('--save_path',   default=None, help='Dir to save adapter weights')
    parser.add_argument('--filename',    default='adapter_weights')
    parser.add_argument('--eval_only',   default=False, action='store_true')
    parser.add_argument('--no_linear_probe', default=False, action='store_true',
                        help='Skip the frozen-backbone linear probe baseline for DINO (saves time)')
    parser.add_argument('--results_csv', default='results/accuracy_results.csv',
                        help='CSV file to append zero-shot and fine-tuned accuracies')

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_arguments()
    set_random_seed(args.seed)

    print(f'\n{"="*60}')
    print(f' Model: {args.model.upper()}  |  Method: {args.method.upper()}')
    print(f' Dataset: {args.dataset}  |  Shots: {args.shots}  |  Seed: {args.seed}')
    print(f'{"="*60}\n')

    clip_preprocess = None
    dataset_preprocess = None

    # ── Build the dataset with model-appropriate default preprocess (only
    #    required by datasets whose constructor consumes a preprocess, e.g. ImageNet)
    if args.dataset == 'imagenet':
        if args.model == 'clip':
            clip_model_ref, clip_preprocess = clip.load(args.backbone)
            dataset_preprocess = clip_preprocess
            del clip_model_ref  # free memory; will reload inside run_clip()
        elif args.model == 'dino':
            dataset_preprocess = _get_dino_preprocess(training=False)
        elif args.model == 'siglip':
            dataset_preprocess = _get_siglip_preprocess(training=False)
        else:
            dataset_preprocess = _get_align_preprocess(training=False)
    elif args.model == 'clip':
        clip_model_ref, clip_preprocess = clip.load(args.backbone)
        del clip_model_ref  # free memory; will reload inside run_clip()

    # ── Build dataset ──────────────────────────────────────────────────────────
    print('Preparing dataset…')
    dataset = build_dataset(args.dataset, args.root_path, args.shots, dataset_preprocess)

    # ── Build DataLoaders with model-appropriate preprocessing ────────────────
    train_loader, val_loader, test_loader = _make_loaders(args, dataset, clip_preprocess)

    # ── Run the right training routine ────────────────────────────────────────
    lp_acc = None
    if args.model == 'clip':
        zs_acc, ft_acc = run_clip(args, dataset, train_loader, val_loader, test_loader)
    elif args.model == 'dino':
        zs_acc, lp_acc, ft_acc = run_dino(args, dataset, train_loader, val_loader, test_loader)
    elif args.model == 'siglip':
        zs_acc, ft_acc = run_siglip(args, dataset, train_loader, val_loader, test_loader)
    elif args.model == 'align':
        zs_acc, ft_acc = run_align(args, dataset, train_loader, val_loader, test_loader)

    print(f'\n{"="*60}')
    if args.model == 'dino':
        print(f' {"Prototype (k-shot) accuracy":25}: {zs_acc:.2f}%')
        if lp_acc is not None:
            print(f' {"Linear probe accuracy":25}: {lp_acc:.2f}%')
        print(f' {"LoRA+head accuracy":25}: {ft_acc:.2f}%')
        if lp_acc is not None:
            print(f' {"Adapter delta (ft-lp)":25}: {ft_acc - lp_acc:+.2f}%')
    else:
        print(f' {"Zero-shot accuracy":25}: {zs_acc:.2f}%')
        print(f' {"Fine-tuned accuracy":25}: {ft_acc:.2f}%')
        print(f' {"Delta":25}: {ft_acc - zs_acc:+.2f}%')
    print(f'{"="*60}\n')

    _append_results_csv(args, zs_acc, ft_acc, lp_acc=lp_acc)


if __name__ == '__main__':
    main()
