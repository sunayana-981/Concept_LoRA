from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.nn import utils
from utils import collect_clip_embeddings, standardize
import torch
import os
from utils import get_optimizer, get_scheduler


def train_sae(args, sae_model, train_loader, mean, std, device, epochs=10, lr=3e-4, l2_decay=1e-6):
    sae_model.to(device).train()
    opt = get_optimizer(sae_model, args)
    for ep in range(epochs):
        tot = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            enc, dec = sae_model(batch)
            loss = sae_model.loss_function(dec, batch, enc)  # your KL sparsity + MSE
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sae_model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs} | Loss {tot/len(train_loader):.4f}")

    model_path = os.path.join(
                args.save_model_dir,
                f"sae_model_epoch_{ep+1}.pt",
            )
    torch.save({"state_dict": sae_model.state_dict(), "mean": mean, "std": std}, model_path)
    print(f"Saved to {model_path}")
    

def preprocess(model, loader, device, batch_tokens=4096):
    embeddings = collect_clip_embeddings(model, loader, device)
    std_embeddings, mean, std = standardize(embeddings)
    dl = DataLoader(TensorDataset(std_embeddings), batch_size=4096, shuffle=True, pin_memory=True)
    return dl, mean, std, std_embeddings.shape[1]