"""
Visual Prompt Tuning (VPT-Deep, Jia et al. 2022) few-shot fine-tuning.

Vision-only technique -- applies to CLIP, DINOv2, and SigLIP2's vision towers
(all standard ViT-style transformers). Excludes ALIGN: its EfficientNet
vision tower has no patch-token sequence for prompts to be concatenated into
(the same structural reason train_maple.py's MaPLe excludes it).

Trains (1) a set of learnable prompt tokens injected at every transformer
block (VPTVisionWrapper, patchsae/src/models/architecture/vpt.py) and (2) a
linear classification head on the pooled visual representation; the backbone
itself stays entirely frozen. This is deliberately closer to the original
VPT paper's linear-probe methodology than to CLIP-style zero-shot text
matching -- it needs no text tower, which makes it the one fine-tuning
technique in this repo that applies uniformly to DINOv2 as well as CLIP and
SigLIP2.

Usage:
    python train_vpt.py --model clip --dataset eurosat --root_path ./data --shots 16 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

_patchsae_root = str(Path(__file__).resolve().parent / "patchsae")
if _patchsae_root not in sys.path:
    sys.path.insert(0, _patchsae_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from CLIP_LoRA.datasets import build_dataset
from CLIP_LoRA.datasets.utils import build_data_loader

from src.models.architecture.vpt import VPTVisionWrapper
from src.models.utils import get_base_backbone
from src.sae_training.backbone_registry import get_backbone_spec

# Matches unified_finetune.py's per-backbone preprocessing constants.
PREPROCESS_STATS = {
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "dino": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "siglip2": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def build_transforms(arch: str, image_size: int = 224):
    mean, std = PREPROCESS_STATS[arch]
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfm, test_tfm


def get_vision_tower(model, arch: str):
    if arch in ("clip", "siglip2"):
        return model.vision_model
    return model  # dino: Dinov2Model IS the vision tower, no wrapper attribute


def get_pooled_output(vision_out, arch: str):
    if getattr(vision_out, "pooler_output", None) is not None:
        return vision_out.pooler_output
    return vision_out.last_hidden_state[:, 0]


@torch.no_grad()
def evaluate(wrapped, head, loader, arch, device) -> float:
    wrapped.eval()
    head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = wrapped(pixel_values=images)
        logits = head(get_pooled_output(out, arch))
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    wrapped.train()
    head.train()
    return 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, choices=["clip", "dino", "siglip2"])
    parser.add_argument("--model_name", type=str, default=None,
                        help="HF model id. Defaults to the --model's registered default.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_prompt_tokens", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Prompt dropout. The official VPT recipe tunes this per-dataset "
                             "(commonly 0.1); left at 0.0 here as a conservative default.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_iters", type=int, default=50,
                        help="total_steps = n_iters * shots, matching unified_finetune.py's convention.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="unified_weights")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    spec = get_backbone_spec(args.model)
    model_id = args.model_name or spec.default_model_id

    print(f"[VPT] Loading base {args.model} ({model_id})...")
    base_model, _ = get_base_backbone(args.model, model_id)
    base_model = base_model.to(device)
    vision_tower = get_vision_tower(base_model, args.model)
    for p in vision_tower.parameters():
        p.requires_grad_(False)

    num_layers = vision_tower.config.num_hidden_layers
    hidden_dim = vision_tower.config.hidden_size
    patch_size = vision_tower.config.patch_size
    # backbone_registry's block_attr_template is relative to the *full*
    # model (e.g. "vision_model.encoder.layers[{i}]" for clip/siglip2), but
    # we've already extracted the vision submodule above -- rebase the
    # template onto it (a no-op for dino, whose Dinov2Model IS its own
    # vision tower and whose template has no "vision_model." prefix).
    block_attr_template = spec.block_attr_template
    if block_attr_template.startswith("vision_model."):
        block_attr_template = block_attr_template[len("vision_model."):]
    wrapped = VPTVisionWrapper(
        vision_tower, block_attr_template, num_layers, hidden_dim,
        n_prompt_tokens=args.n_prompt_tokens,
        dropout=args.dropout,
        patch_size=patch_size,
    ).to(device)

    print(f"[VPT] Loading dataset '{args.dataset}' ({args.shots} shots, seed {args.seed})...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)
    train_tfm, test_tfm = build_transforms(args.model)
    train_loader = build_data_loader(
        data_source=dataset.train_x, batch_size=args.batch_size,
        tfm=train_tfm, is_train=True, shuffle=True, num_workers=4,
    )
    test_loader = build_data_loader(
        data_source=dataset.test, batch_size=args.eval_batch_size,
        tfm=test_tfm, is_train=False, shuffle=False, num_workers=4,
    )

    num_classes = len(dataset.classnames)
    head = nn.Linear(hidden_dim, num_classes).to(device)

    trainable_params = list(wrapped.prompts.parameters()) + list(head.parameters())
    print(f"[VPT] Trainable params (prompts + head): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    total_iters = max(args.n_iters * args.shots, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    zs_acc = evaluate(wrapped, head, test_loader, args.model, device)
    print(f"[VPT] Pre-training (random prompts + head) test accuracy: {zs_acc:.2f}%")

    print(f"[VPT] Training for {total_iters} steps...")
    t_start = time.time()
    wrapped.train()
    head.train()
    step = 0
    while step < total_iters:
        for images, labels in train_loader:
            if step >= total_iters:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = wrapped(pixel_values=images)
            logits = head(get_pooled_output(out, args.model))
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 20 == 0:
                print(f"[VPT] step {step}/{total_iters}  loss={loss.item():.4f}")

    ft_acc = evaluate(wrapped, head, test_loader, args.model, device)
    elapsed = time.time() - t_start
    print(f"[VPT] Final test accuracy: {ft_acc:.2f}% (pre-training was {zs_acc:.2f}%), elapsed {elapsed:.1f}s")

    save_dir = Path(args.save_path) / f"{args.model}_vpt" / args.dataset / f"{args.shots}shots" / f"seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "vpt_weights.pt"
    torch.save(
        {
            "prompts": wrapped.prompts.state_dict(),
            "head": head.state_dict(),
            "n_prompt_tokens": args.n_prompt_tokens,
            "test_accuracy": ft_acc,
            "pretraining_accuracy": zs_acc,
        },
        save_path,
    )
    print(f"[VPT] Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
