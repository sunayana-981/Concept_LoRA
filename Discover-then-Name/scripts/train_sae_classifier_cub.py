import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
from tqdm import tqdm
import clip
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from sparse_autoencoder import SparseAutoencoder


# -----------------------
# 1. Classifier Module
# -----------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# -----------------------
# 2. Train Classifier
# -----------------------
def train_classifier(features, labels, input_dim, num_classes, device, lr=1e-3, epochs=20, batch_size=256, log_prefix=""):
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Classifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_log = []
    writer = SummaryWriter(log_dir=f"logs/tensorboard/{log_prefix}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        epoch_loss = total_loss / len(dataset)
        loss_log.append((epoch + 1, epoch_loss))
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"[{log_prefix}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    writer.close()

    # Save loss to CSV
    os.makedirs("logs", exist_ok=True)
    df = pd.DataFrame(loss_log, columns=["epoch", "loss"])
    df.to_csv(f"logs/{log_prefix}_train_loss.csv", index=False)

    return model


# -----------------------
# 3. Load CUB Data
# -----------------------
def get_cub_dataloader(data_root, split='train', batch_size=64, image_size=224, num_workers=4):
    image_dir = os.path.join(data_root, "images")
    image_txt = os.path.join(data_root, "images.txt")
    split_txt = os.path.join(data_root, "train_test_split.txt")

    image_df = pd.read_csv(image_txt, sep=' ', header=None, names=['img_id', 'img_path'])
    split_df = pd.read_csv(split_txt, sep=' ', header=None, names=['img_id', 'is_train'])

    is_train = int(split == 'train')
    split_ids = split_df[split_df['is_train'] == is_train]['img_id'].values
    split_img_paths = image_df[image_df['img_id'].isin(split_ids)]['img_path'].tolist()

    full_dataset = ImageFolder(image_dir, transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

    # Use relative paths for mapping
    img_path_to_idx = {
        os.path.relpath(path, image_dir): idx
        for idx, (path, _) in enumerate(full_dataset.samples)
    }

    selected_indices = []
    missing_paths = []

    for rel_path in split_img_paths:
        norm_path = os.path.normpath(rel_path.strip())
        if norm_path in img_path_to_idx:
            selected_indices.append(img_path_to_idx[norm_path])
        else:
            missing_paths.append(norm_path)

    if missing_paths:
        print("⚠️ WARNING: Some paths were not found in ImageFolder samples!")
        print(f"Missing example: {missing_paths[0]}")
        print(f"Total missing: {len(missing_paths)} / {len(split_img_paths)}")
        raise KeyError("Some image paths from metadata were not found in the dataset. Check dataset structure.")

    subset = Subset(full_dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    return loader


# -----------------------
# 4. Extract Features
# -----------------------
def extract_activations(model, dataloader, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model.encode_image(images)
            all_features.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# -----------------------
# 5. Evaluation & Logging
# -----------------------
def evaluate_classifier(model, sae, features, labels, input_type, device):
    model.eval()
    sae.eval()
    features, labels = features.to(device), labels.to(device)

    with torch.no_grad():
        latents, recons = sae(features)
        inputs = latents if input_type == 'latent' else recons
        inputs = inputs.squeeze()
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

    acc = accuracy_score(labels.cpu(), preds.cpu())
    f1 = f1_score(labels.cpu(), preds.cpu(), average='macro')

    print(f"\n📊 Evaluation ({input_type}):")
    print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(classification_report(labels.cpu(), preds.cpu(), digits=4))

    return preds.cpu(), acc, f1


def save_predictions(preds, labels, fname):
    df = pd.DataFrame({
        "ground_truth": labels.numpy(),
        "prediction": preds.numpy()
    })
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    df.to_csv(fname, index=False)
    print(f"📄 Saved predictions to {fname}")


# -----------------------
# 6. Train on SAE Outputs
# -----------------------
def train_on_sae(args, features, labels):
    device = args.device
    features, labels = features.to(device), labels.to(device)

    input_dim = 512      # CLIP ViT-B/16
    latent_dim = 4096    # Matches pretrained SAE
    sae = SparseAutoencoder(n_input_features=input_dim, n_learned_features=latent_dim, n_components=len(args.hook_points)).to(device)

    ckpt_path = "/home/sunayana/Documents/Concept_LoRA/Discover-then-Name/pretrained/Checkpoints/clip_ViT-B:16_sparse_autoencoder_final.pt"
    print(f"🔁 Loading SAE from: {ckpt_path}")
    sae.load_state_dict(torch.load(ckpt_path, map_location=device))
    sae.eval()

    with torch.no_grad():
        latents, recons = sae(features)
        latents, recons = latents.squeeze(), recons.squeeze()

    num_classes = len(labels.unique())

    print("\n🎯 Training classifier on SAE latent space...")
    clf_latent = train_classifier(latents, labels, input_dim=latent_dim, num_classes=num_classes, device=device, log_prefix="latent")

    print("\n🎯 Training classifier on SAE reconstructions...")
    clf_recon = train_classifier(recons, labels, input_dim=input_dim, num_classes=num_classes, device=device, log_prefix="recon")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(clf_latent.state_dict(), os.path.join("saved_models", "clf_latent.pth"))
    torch.save(clf_recon.state_dict(), os.path.join("saved_models", "clf_recon.pth"))
    print("✅ Saved classifiers to 'saved_models/'.")

    return clf_latent, clf_recon, sae


# -----------------------
# 7. Main
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_root", type=str, default="/home/sunayana/Documents/Concept_LoRA/datasets/cub2002011")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modality", type=str, default="img")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hook_points", nargs='+', default=["layer3.5"])
    args = parser.parse_args()

    # Load CLIP
    print("📦 Loading CLIP (ViT-B/16)...")
    clip_model, preprocess = clip.load("ViT-B/16", device=args.device)
    clip_model.eval()

    # Load CUB data
    print("📂 Loading CUB dataset...")
    train_loader = get_cub_dataloader(args.cub_root, split='train', batch_size=args.batch_size)
    test_loader = get_cub_dataloader(args.cub_root, split='test', batch_size=args.batch_size)

    # Extract features
    print("🔍 Extracting train features...")
    train_features, train_labels = extract_activations(clip_model, train_loader, args.device)
    print("🔍 Extracting test features...")
    test_features, test_labels = extract_activations(clip_model, test_loader, args.device)

    # Train classifiers
    print("🚀 Training classifiers on SAE representations...")
    clf_latent, clf_recon, sae = train_on_sae(args, train_features, train_labels)

    # Evaluate on test set
    print("\n🔬 Evaluating on test set...")
    preds_latent, acc_latent, f1_latent = evaluate_classifier(clf_latent, sae, test_features, test_labels, input_type='latent', device=args.device)
    save_predictions(preds_latent, test_labels, "logs/predictions_latent.csv")

    preds_recon, acc_recon, f1_recon = evaluate_classifier(clf_recon, sae, test_features, test_labels, input_type='recon', device=args.device)
    save_predictions(preds_recon, test_labels, "logs/predictions_recon.csv")

    print("\n🏁 Done! Results:")
    print(f"Latent Classifier - Acc: {acc_latent:.4f}, F1: {f1_latent:.4f}")
    print(f"Recon Classifier  - Acc: {acc_recon:.4f}, F1: {f1_recon:.4f}")
    print("\n📊 Logs available in 'logs/' and TensorBoard: 'logs/tensorboard/'")
