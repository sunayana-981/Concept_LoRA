"""
MaPLe (Multi-modal Prompt Learning, Khattak et al. 2023) fine-tuning.

No training loop for MaPLe existed anywhere in this repo before this --
patchsae/src/models/architecture/maple.py's CustomCLIP is a complete, correct
port of the official architecture, but the rest of the repo only ever *loads*
already-trained MaPLe checkpoints (for downstream SAE feature extraction via
get_adapted_clip(model_type="maple", ...)). This script actually trains the
MultiModalPromptLearner's prompts end-to-end on a target dataset, following
the official recipe's hyperparameters (SGD, cosine schedule, 1-epoch constant
warmup), and saves a checkpoint in the exact format that loader already
expects -- so a checkpoint trained here needs no downstream changes to use.

Scope: CLIP only. MaPLe's mechanism -- per-transformer-block deep visual
prompts coupled to the text branch via a learned projection -- is implemented
against this repo's CLIP-specific fork (src/models/clip/). Porting it to
ALIGN (EfficientNet vision tower, no per-block token sequence to inject
prompts into) is architecturally impossible; porting it to SigLIP2 (different
transformer internals and a sigmoid rather than contrastive-softmax loss)
would be a new research contribution, not a plumbing extension -- so neither
is attempted here.

Usage:
    python train_maple.py --dataset eurosat --root_path ./data --shots 16 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent)
_patchsae_root = str(Path(__file__).resolve().parent / "patchsae")
for _p in (_repo_root, _patchsae_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from CLIP_LoRA.datasets import build_dataset
from CLIP_LoRA.datasets.utils import build_data_loader

from src.models.architecture.maple import CustomCLIP
from src.models.config.maple import get_maple_config
from src.models.utils import load_clip_model

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(image_size: int):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return train_tfm, test_tfm


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    text_features = model.get_text_features()
    logit_scale = model.logit_scale.exp()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        image_features = model(None, None, images)
        logits = logit_scale * image_features @ text_features.t()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return 100.0 * correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--config_path", type=str,
        default="patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Overrides the yaml's OPTIM.MAX_EPOCH.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Overrides the yaml's DATALOADER.TRAIN_X.BATCH_SIZE.")
    parser.add_argument("--save_path", type=str, default="maple_weights")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = get_maple_config(custom_clip_cfg=args.config_path)
    if args.epochs is not None:
        cfg.OPTIM.MAX_EPOCH = args.epochs
    if args.batch_size is not None:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.freeze()

    print(f"[MaPLe] Loading dataset '{args.dataset}' ({args.shots} shots, seed {args.seed})...")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)
    train_tfm, test_tfm = build_transforms(cfg.INPUT.SIZE[0])
    train_loader = build_data_loader(
        data_source=dataset.train_x, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        tfm=train_tfm, is_train=True, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    test_loader = build_data_loader(
        data_source=dataset.test, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        tfm=test_tfm, is_train=False, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    print("[MaPLe] Building CustomCLIP...")
    # MultiModalPromptLearner.__init__ tokenizes CTX_INIT and looks up
    # embeddings via clip_model.token_embedding on CPU tensors -- build on
    # CPU first, then move the fully-assembled model to device, rather than
    # moving clip_model to device before CustomCLIP wraps it.
    clip_model = load_clip_model(cfg, "maple").float()
    model = CustomCLIP(cfg, dataset.classnames, clip_model).to(device)

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("prompt_learner."))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[MaPLe] Trainable params (prompt_learner only): "
          f"{sum(p.numel() for p in trainable_params):,}")

    zs_acc = evaluate(model, test_loader, device)
    print(f"[MaPLe] Zero-shot (untrained prompts) test accuracy: {zs_acc:.2f}%")

    max_epoch = cfg.OPTIM.MAX_EPOCH
    warmup_epoch = cfg.OPTIM.WARMUP_EPOCH
    optimizer = torch.optim.SGD(trainable_params, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epoch - warmup_epoch, 1)
    )
    # Matches the official Dassl trainer's own PREC="fp16" path exactly:
    # autocast + GradScaler over an fp32 base model, not manual .half()
    # conversion (confirmed against trainers/maple.py's forward_backward).
    use_amp = (cfg.TRAINER.MAPLE.PREC == "fp16") and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[MaPLe] Training for {max_epoch} epochs (SGD lr={cfg.OPTIM.LR}, "
          f"{warmup_epoch}-epoch constant warmup @ {cfg.OPTIM.WARMUP_CONS_LR})...")
    model.train()
    t_start = time.time()
    for epoch in range(max_epoch):
        if epoch < warmup_epoch:
            for g in optimizer.param_groups:
                g["lr"] = cfg.OPTIM.WARMUP_CONS_LR
        running_loss, n_batches = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=use_amp):
                image_features = model(None, None, images)
                text_features = model.get_text_features()
                logits = model.logit_scale.exp() * image_features @ text_features.t()
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
        if epoch >= warmup_epoch:
            main_scheduler.step()
        print(f"[MaPLe] epoch {epoch + 1}/{max_epoch}  loss={running_loss / max(n_batches, 1):.4f}")

    ft_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - t_start
    print(f"[MaPLe] Final test accuracy: {ft_acc:.2f}% (zero-shot was {zs_acc:.2f}%), elapsed {elapsed:.1f}s")

    config_name = Path(args.config_path).stem
    save_dir = Path(args.save_path) / args.dataset / "base" / f"seed{args.seed}" / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"model.pth.tar-{max_epoch}"
    # Save only prompt_learner's params (~14MB), not the full CustomCLIP
    # state_dict (~570MB, since it also includes the frozen, unchanged CLIP
    # image_encoder/text_encoder). Safe: the downstream loader
    # (get_adapted_clip -> load_state_dict_without_prompt_learner) always
    # calls model.load_state_dict(..., strict=False), so missing
    # encoder keys just keep the freshly-loaded pretrained values it already
    # has -- identical to what re-saving them here would produce anyway.
    prompt_learner_state = {
        f"prompt_learner.{k}": v for k, v in model.prompt_learner.state_dict().items()
    }
    torch.save(
        {"state_dict": prompt_learner_state, "epoch": max_epoch,
         "test_accuracy": ft_acc, "zero_shot_accuracy": zs_acc},
        save_path,
    )
    print(f"[MaPLe] Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
