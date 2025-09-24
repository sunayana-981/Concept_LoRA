import torch, torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    LambdaLR,
    CyclicLR,
    CosineAnnealingLR,
    StepLR,
    LinearLR,
    ConstantLR,
)

@torch.no_grad()
def collect_clip_embeddings(model, dl, device):
    embs = []
    for imgs in dl:
        imgs = imgs.to(device, non_blocking=True)
        x = model.encode_image(imgs)          # [B, Dproj] (e.g., Dproj=512 for ViT-B-32)
        x = F.normalize(x, dim=-1)            # already normalized by CLIP, but keep it safe
        embs.append(x.cpu())
    X = torch.cat(embs, dim=0)                # [N, Dproj]
    return X

def standardize(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X - mean) / std, mean, std


def get_scheduler(opt, args, trainloader):
    if args.scheduler == "onecycle":
        scheduler = CyclicLR(
            opt,
            base_lr=1e-5,
            max_lr=args.lr,
            step_size_up=2 * len(trainloader),
            mode="triangular2",
        )
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            opt, T_max=args.epochs * len(trainloader), eta_min=1e-5
        )
    elif args.scheduler == "step":
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
    elif args.scheduler == "linear":
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=args.epochs * len(trainloader),
        )
    elif args.scheduler == "constant":
        scheduler = ConstantLR(
            opt, factor=1, total_iters=args.epochs * len(trainloader), last_epoch=-1
        )
    elif args.scheduler == "none":
        scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError(f"Unknown scheduler {args.scheduler}")
    return scheduler


def get_optimizer(model, args):
    if args.opt == "adamw":
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        opt = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return opt