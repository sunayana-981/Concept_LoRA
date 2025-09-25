import random
import argparse
import torch
import numpy as np
import os
import json
from datetime import datetime
from dataloader import get_dataloader
from model import get_model
from train import train_sae, preprocess

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args, logger=None):
    if args.dataset == "mscoco":
        images_dir = "/data1/ai22resch11001/projects/data/mscoco/train2017"
        annotations_file = "/data1/ai22resch11001/projects/data/mscoco/annotations/instances_train2017.json"

    _, dataloader = get_dataloader(args.dataset, images_dir, annotations_file, subset=1, batch_size=args.batch_size)
    base_model = get_model(args.base_model_name) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    processed_loader, mean, std, d_in = preprocess(base_model, dataloader, device, batch_tokens=4096)

    sae_model = get_model("sae", d_in=d_in)
    train_sae(args, sae_model, processed_loader, mean, std, device, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concept-based LoRA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--base_model_name", type=str, default="clip_vit-b-32", help="Pretrained model name or path")
    parser.add_argument(
        "--dataset", type=str, default="mscoco", help="Dataset name"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./saved_models/",
        help="Path to save models",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        help="Optimizer type",
        choices=["adam", "sgd", "adamw"],
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        help="Scheduler type",
        choices=["cosine", "step", "linear", "onecycle", "constant", "none"],
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-5, help="Weight decay for optimizer"
    )

    args = parser.parse_args()

    save_model_dir = os.path.join(
        args.save_model_dir,
        args.dataset,
        f"exp_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}",
    )
    args.save_model_dir = save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    # save the arguments to a json file
    with open(os.path.join(save_model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    set_seed(args.seed)
    main(args)