# train_clip_lora_cub.py
# pip install open_clip_torch torchvision torch matplotlib
import os, argparse, time, math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import open_clip
from open_clip import create_model_and_transforms

# ---------------------------
# LoRA wrapper for Linear
# ---------------------------
class LoRAWrappedLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.in_features, self.out_features = in_f, out_f

        # --- register frozen base weights directly on this module ---
        w = base_linear.weight.detach()
        self.weight = nn.Parameter(w, requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # --- LoRA params ---
        self.r = int(r)
        self.scaling = (alpha / r) if r > 0 else 0.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_f))
            self.lora_B = nn.Parameter(torch.zeros(out_f, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        # base linear (frozen)
        out = F.linear(x, self.weight, self.bias)
        # LoRA delta
        if self.r and self.lora_A is not None:
            lora = x @ self.lora_A.t()
            lora = self.dropout(lora)
            lora = lora @ self.lora_B.t()
            out = out + self.scaling * lora
        return out

    def extra_repr(self):  
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, scaling={self.scaling}")


def loraize_linears(module: nn.Module, name_filter: List[str], r: int, alpha: int, dropout: float):
    """
    Replace nn.Linear modules under `module` with LoRAWrappedLinear if their qualified name
    contains any of strings in `name_filter`. If name_filter is empty, apply to all.
    """
    # Walk with named_modules to get qualified names
    replace_map = []
    for qname, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            if (len(name_filter) == 0) or any(s in qname for s in name_filter):
                replace_map.append(qname)

    # Replace by navigating parents
    for qname in replace_map:
        parent, attr = _resolve_parent_and_attr(module, qname)
        base_linear = getattr(parent, attr)
        setattr(parent, attr, LoRAWrappedLinear(base_linear, r=r, alpha=alpha, dropout=dropout))

def _resolve_parent_and_attr(root: nn.Module, qname: str) -> Tuple[nn.Module, str]:
    parts = qname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

# ---------------------------
# Classifier Head
# ---------------------------
class CLIPLinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# ---------------------------
# Utils
# ---------------------------
@torch.no_grad()
def evaluate(model, head, loader, device, amp_dtype):
    model.eval(); head.eval()
    n, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
            img_feats = model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1)
            logits = head(img_feats)
        pred = logits.argmax(dim=1)
        n += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / n

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of CLIP (vision tower) on CUB-200-2011")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to CUB images root (e.g., .../CUB_200_2011/images)")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--targets", type=str, default="attn,mlp",
                        help="Comma list of substrings to select linears (e.g., 'attn,mlp'). Empty=all linears.")
    parser.add_argument("--save", type=str, default="cub_clip_lora.pt")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training set used for val if no separate split.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16  # autocast dtype

    # ---- Data (CUB is class-subfoldered under images/) ----
    # Train/val split from full dataset if you haven't created splits.
    # If you already have separate folders, point --data to train and modify this block.
    full = datasets.ImageFolder(args.data)
    num_classes = len(full.classes)
    n_total = len(full)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val], generator=gen)

    # CLIP transforms
    _, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    train_set.dataset.transform = preprocess_train
    val_set.dataset.transform = preprocess_val

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- Model ----
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()  # we'll freeze base (text & vision) except LoRA-injected linears

    # Inject LoRA into the vision tower linear layers
    target_substrings = [t.strip() for t in args.targets.split(",") if t.strip() != ""]
    model.visual.to(device) 
    loraize_linears(model.visual, target_substrings, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # Enable LoRA params in visual only
    for n, p in model.visual.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    # Classifier head on CLIP image features
    # embed_dim is model.visual.output_dim for CLIP image features
    embed_dim = model.visual.output_dim
    head = CLIPLinearHead(embed_dim, num_classes).to(device)

    # Optimizer
    params = list(p for p in model.visual.parameters() if p.requires_grad) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Train
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    print(f"Trainable params: LoRA+head = {count_trainable_params(model.visual) + count_trainable_params(head):,}")

    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        epoch_loss, n_seen = 0.0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                img_feats = model.encode_image(images)            # [B, D]
                img_feats = F.normalize(img_feats, dim=-1)
                logits = head(img_feats)                          # [B, C]
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            n_seen += bs
            epoch_loss += loss.item() * bs

        train_loss = epoch_loss / n_seen
        val_acc = evaluate(model, head, val_loader, device, amp_dtype)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | val@1 {val_acc:.2f}% | {dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            # Save LoRA and head weights only
            save_obj = {
                "model_name": args.model,
                "pretrained": args.pretrained,
                "lora_visual": {k: v.cpu() for k, v in model.visual.state_dict().items() if "lora_" in k},
                "head": head.state_dict(),
                "class_to_idx": full.class_to_idx,
            }
            torch.save(save_obj, args.save)
            print(f"  ✓ Saved best to {args.save} (val@1={best_acc:.2f}%)")

    print(f"Done. Best val@1: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
