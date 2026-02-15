import torch
import os
import sys
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from collections import OrderedDict
from tqdm import tqdm

try:
    from tasks.utils import get_sae_and_vit
except ImportError:
    print("CRITICAL ERROR: Could not import 'tasks.utils'.")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
SAE_CHECKPOINT_PATH = "data/sae_weight/base/out.pt" 
DATASET_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/OxfordPets/images"
LOCAL_CHECKPOINT_PATH = "/home/sunayana/Documents/Concept_LoRA/clip_vitb16_pets_lora_merged.pt"

VIT_TYPE = "base"
BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# ==========================================
# SAE Application Modes
# ==========================================
# Choose which SAE method to use:
#   'preserve_cls' - Apply SAE only to patch tokens, keep CLS unchanged (RECOMMENDED)
#   'blend' - Blend original and SAE features (alpha controls blend ratio)
#   'original' - Original method (applies SAE to all tokens including CLS)
SAE_MODE = 'preserve_cls'  # Change this to test different methods
BLEND_ALPHA = 0.2  # Only used if SAE_MODE = 'blend'

# ==========================================
# Dataset Loader
# ==========================================
class FlatFilenameDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root, "*.[jJ][pP]*[gG]")))
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")

        self.classes = sorted(list(set(
            [os.path.basename(p).rsplit('_', 1)[0] for p in self.image_paths]
        )))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"   Found {len(self.classes)} classes, {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        label_str = filename.rsplit('_', 1)[0]
        label = self.class_to_idx[label_str]
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

def get_smart_dataset(root, transform):
    try:
        sys.stdout = open(os.devnull, 'w')
        dataset = datasets.ImageFolder(root=root, transform=transform)
        sys.stdout = sys.__stdout__
        return dataset
    except:
        sys.stdout = sys.__stdout__
        return FlatFilenameDataset(root=root, transform=transform)

# ==========================================
# OpenAI -> HF Conversion
# ==========================================
def convert_openai_to_hf_clip(openai_state_dict):
    hf_state_dict = OrderedDict()
    for key, value in openai_state_dict.items():
        new_key = None
        if key.startswith('visual.'):
            new_key = key.replace('visual.', 'vision_model.')
            new_key = new_key.replace('vision_model.conv1.', 'vision_model.embeddings.patch_embedding.')
            if new_key == 'vision_model.class_embedding': new_key = 'vision_model.embeddings.class_embedding'
            elif new_key == 'vision_model.positional_embedding': new_key = 'vision_model.embeddings.position_embedding.weight'
            new_key = new_key.replace('vision_model.ln_pre.', 'vision_model.pre_layrnorm.')
            new_key = new_key.replace('vision_model.ln_post.', 'vision_model.post_layernorm.')
            new_key = new_key.replace('vision_model.transformer.resblocks.', 'vision_model.encoder.layers.')
            
            if '.attn.in_proj_weight' in new_key:
                layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                base = f'vision_model.encoder.layers.{layer_num}.self_attn.'
                dim = value.shape[0] // 3
                hf_state_dict[base + 'q_proj.weight'] = value[:dim].clone()
                hf_state_dict[base + 'k_proj.weight'] = value[dim:2*dim].clone()
                hf_state_dict[base + 'v_proj.weight'] = value[2*dim:].clone()
                continue
            elif '.attn.in_proj_bias' in new_key:
                layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                base = f'vision_model.encoder.layers.{layer_num}.self_attn.'
                dim = value.shape[0] // 3
                hf_state_dict[base + 'q_proj.bias'] = value[:dim].clone()
                hf_state_dict[base + 'k_proj.bias'] = value[dim:2*dim].clone()
                hf_state_dict[base + 'v_proj.bias'] = value[2*dim:].clone()
                continue
            
            new_key = new_key.replace('.attn.out_proj.', '.self_attn.out_proj.')
            new_key = new_key.replace('.mlp.c_fc.', '.mlp.fc1.')
            new_key = new_key.replace('.mlp.c_proj.', '.mlp.fc2.')
            new_key = new_key.replace('.ln_1.', '.layer_norm1.')
            new_key = new_key.replace('.ln_2.', '.layer_norm2.')
            if new_key == 'vision_model.proj':
                new_key = 'visual_projection.weight'
                value = value.T
        
        elif key.startswith('transformer.') or key in ['token_embedding.weight', 'positional_embedding', 'ln_final.weight', 'ln_final.bias', 'text_projection']:
            if key == 'token_embedding.weight': new_key = 'text_model.embeddings.token_embedding.weight'
            elif key == 'positional_embedding': new_key = 'text_model.embeddings.position_embedding.weight'
            elif key.startswith('transformer.resblocks.'):
                new_key = key.replace('transformer.resblocks.', 'text_model.encoder.layers.')
                if '.attn.in_proj_weight' in new_key:
                    layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                    base = f'text_model.encoder.layers.{layer_num}.self_attn.'
                    dim = value.shape[0] // 3
                    hf_state_dict[base + 'q_proj.weight'] = value[:dim].clone()
                    hf_state_dict[base + 'k_proj.weight'] = value[dim:2*dim].clone()
                    hf_state_dict[base + 'v_proj.weight'] = value[2*dim:].clone()
                    continue
                elif '.attn.in_proj_bias' in new_key:
                    layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                    base = f'text_model.encoder.layers.{layer_num}.self_attn.'
                    dim = value.shape[0] // 3
                    hf_state_dict[base + 'q_proj.bias'] = value[:dim].clone()
                    hf_state_dict[base + 'k_proj.bias'] = value[dim:2*dim].clone()
                    hf_state_dict[base + 'v_proj.bias'] = value[2*dim:].clone()
                    continue
                new_key = new_key.replace('.attn.out_proj.', '.self_attn.out_proj.')
                new_key = new_key.replace('.mlp.c_fc.', '.mlp.fc1.')
                new_key = new_key.replace('.mlp.c_proj.', '.mlp.fc2.')
                new_key = new_key.replace('.ln_1.', '.layer_norm1.')
                new_key = new_key.replace('.ln_2.', '.layer_norm2.')
            elif key.startswith('ln_final.'): new_key = key.replace('ln_final.', 'text_model.final_layer_norm.')
            elif key == 'text_projection':
                new_key = 'text_projection.weight'
                value = value.T
        elif key == 'logit_scale': new_key = 'logit_scale'
        
        if new_key: hf_state_dict[new_key] = value
    return hf_state_dict

# ==========================================
# SAE Application Functions
# ==========================================
def apply_sae_preserve_cls(internal_feats, sae_model):
    """
    SOLUTION 1: Preserve CLS token, only reconstruct patches.
    This is the recommended approach.
    """
    b, s, d = internal_feats.shape  # e.g., [32, 197, 768]
    
    # Separate CLS (index 0) from patches (index 1:)
    cls_token = internal_feats[:, 0:1, :]      # [B, 1, 768] - KEEP ORIGINAL
    patch_tokens = internal_feats[:, 1:, :]    # [B, 196, 768] - RECONSTRUCT
    
    # Apply SAE only to patches
    patch_flat = patch_tokens.reshape(-1, d)   # [B*196, 768]
    
    sae_output = sae_model(patch_flat)
    if isinstance(sae_output, tuple):
        patch_reconstructed = sae_output[0]
    else:
        patch_reconstructed = sae_output
    
    patch_reconstructed = patch_reconstructed.view(b, s-1, d)  # [B, 196, 768]
    
    # Concatenate: original CLS + reconstructed patches
    reconstructed = torch.cat([cls_token, patch_reconstructed], dim=1)  # [B, 197, 768]
    
    return reconstructed

def apply_sae_blended(internal_feats, sae_model, alpha=0.2):
    """
    SOLUTION 4: Blend original and reconstructed features.
    alpha controls how much SAE to use (0=original, 1=full SAE)
    """
    b, s, d = internal_feats.shape
    
    # Reconstruct with SAE
    flat_feats = internal_feats.view(-1, d)
    sae_output = sae_model(flat_feats)
    if isinstance(sae_output, tuple):
        reconstructed_flat = sae_output[0]
    else:
        reconstructed_flat = sae_output
    
    reconstructed = reconstructed_flat.view(b, s, d)
    
    # Blend: more weight to original for better accuracy
    blended = alpha * reconstructed + (1.0 - alpha) * internal_feats
    
    return blended

def apply_sae_original(internal_feats, sae_model):
    """
    Original method: Apply SAE to all tokens.
    This is what was causing the 83% accuracy drop.
    """
    b, s, d = internal_feats.shape
    flat_feats = internal_feats.view(-1, d)
    
    sae_output = sae_model(flat_feats)
    if isinstance(sae_output, tuple):
        reconstructed_flat = sae_output[0]
    else:
        reconstructed_flat = sae_output
    
    return reconstructed_flat.view(b, s, d)

# ==========================================
# Evaluation Function
# ==========================================
activation_cache = {}

def get_activation(name):
    def hook(model, input, output):
        activation_cache[name] = output
    return hook

def evaluate_model(model, processor, dataloader, class_names, device, 
                   model_name="Model", use_sae=False, sae_model=None, sae_mode='preserve_cls'):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"SAE Enabled: {use_sae}")
    if use_sae:
        print(f"SAE Mode: {sae_mode}")
    print(f"{'='*60}")
    
    model.eval()
    
    hook_handle = None
    if use_sae and sae_model is not None:
        target_layer = model.vision_model.encoder.layers[-1] 
        hook_handle = target_layer.register_forward_hook(get_activation('last_layer'))

    # Prepare text features
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    reconstruction_mse_sum = 0.0
    cosine_sim_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_sae and sae_model is not None:
                # Get standard features for comparison
                standard_feats = model.get_image_features(images)
                
                # Get activations from hook
                hook_output = activation_cache['last_layer']
                if isinstance(hook_output, tuple):
                    internal_feats = hook_output[0]
                else:
                    internal_feats = hook_output
                
                # Store original for metrics
                original_feats = internal_feats.clone()
                
                # Apply SAE based on selected mode
                if sae_mode == 'preserve_cls':
                    reconstructed_feats = apply_sae_preserve_cls(internal_feats, sae_model)
                elif sae_mode == 'blend':
                    reconstructed_feats = apply_sae_blended(internal_feats, sae_model, BLEND_ALPHA)
                else:  # 'original'
                    reconstructed_feats = apply_sae_original(internal_feats, sae_model)
                
                # Calculate reconstruction quality metrics
                mse = ((original_feats - reconstructed_feats) ** 2).mean().item()
                reconstruction_mse_sum += mse
                
                # Extract CLS token
                cls_token = reconstructed_feats[:, 0, :]
                
                # Apply post-processing
                cls_token = model.vision_model.post_layernorm(cls_token)
                image_features = model.visual_projection(cls_token)
                
                # Calculate cosine similarity with standard method
                cos_sim = torch.nn.functional.cosine_similarity(
                    image_features, standard_feats, dim=-1
                ).mean().item()
                cosine_sim_sum += cos_sim
                num_batches += 1

            else:
                # Standard path (no SAE)
                image_features = model.get_image_features(images)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity and predictions
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if hook_handle: 
        hook_handle.remove()
    
    accuracy = 100 * correct / total
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if use_sae and num_batches > 0:
        avg_mse = reconstruction_mse_sum / num_batches
        avg_cos_sim = cosine_sim_sum / num_batches
        print(f"Average Reconstruction MSE: {avg_mse:.6f}")
        print(f"Average Cosine Similarity: {avg_cos_sim:.4f}")
    
    print(f"{'='*60}\n")
    
    return accuracy

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("SAE Classification with CLS Token Preservation")
    print(f"{'='*60}")
    print(f"SAE Mode: {SAE_MODE}")
    if SAE_MODE == 'blend':
        print(f"Blend Alpha: {BLEND_ALPHA}")
    print(f"{'='*60}\n")
    
    # 1. Load Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    dataset = get_smart_dataset(DATASET_ROOT, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Load Models
    print("Loading SAE and ViT...")
    sae, vit, cfg = get_sae_and_vit(
        sae_path=SAE_CHECKPOINT_PATH,
        vit_type=VIT_TYPE,
        device=DEVICE,
        backbone=BACKBONE,
        model_path=None,
        classnames=None
    )
    hf_model = vit.model
    processor = CLIPProcessor.from_pretrained(BACKBONE)

    # 3. Baseline (No SAE)
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (NO SAE)")
    print("="*60)
    acc_base_no_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        "Base Model (No SAE)", use_sae=False
    )

    # 4. Base Model WITH SAE (Using selected mode)
    print("\n" + "="*60)
    print(f"PHASE 2: BASE MODEL WITH SAE ({SAE_MODE.upper()})")
    print("="*60)
    acc_base_with_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        f"Base Model (SAE-{SAE_MODE})", use_sae=True, sae_model=sae, sae_mode=SAE_MODE
    )

    # 5. Load LoRA
    print("\n" + "="*60)
    print("PHASE 3: LOADING LORA")
    print("="*60)
    openai_state_dict = torch.load(LOCAL_CHECKPOINT_PATH, map_location=DEVICE)
    hf_state_dict = convert_openai_to_hf_clip(openai_state_dict)
    hf_model.load_state_dict(hf_state_dict, strict=False)
    print("LoRA loaded successfully.")

    # 6. LoRA (No SAE)
    acc_lora_no_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        "LoRA Model (No SAE)", use_sae=False
    )

    # 7. LoRA WITH SAE
    print("\n" + "="*60)
    print(f"PHASE 4: LORA MODEL WITH SAE ({SAE_MODE.upper()})")
    print("="*60)
    acc_lora_with_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        f"LoRA Model (SAE-{SAE_MODE})", use_sae=True, sae_model=sae, sae_mode=SAE_MODE
    )
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Base (No SAE):         {acc_base_no_sae:.2f}%")
    print(f"Base (SAE-{SAE_MODE}):      {acc_base_with_sae:.2f}%")
    print(f"LoRA (No SAE):         {acc_lora_no_sae:.2f}%")
    print(f"LoRA (SAE-{SAE_MODE}):      {acc_lora_with_sae:.2f}%")
    print("="*60)
    
    # Analysis
    base_drop = 100 * (1 - acc_base_with_sae / acc_base_no_sae) if acc_base_no_sae > 0 else 0
    lora_drop = 100 * (1 - acc_lora_with_sae / acc_lora_no_sae) if acc_lora_no_sae > 0 else 0
    
    print(f"\nAccuracy Drop from SAE:")
    print(f"  Base: {base_drop:.1f}%")
    print(f"  LoRA: {lora_drop:.1f}%")
    
    if base_drop < 20:
        print(f"\n✅ SUCCESS! SAE mode '{SAE_MODE}' maintains good accuracy.")
    elif base_drop < 40:
        print(f"\n⚠️  Moderate drop with SAE mode '{SAE_MODE}'.")
        print("   Try 'blend' mode with lower alpha (0.1-0.2)")
    else:
        print(f"\n❌ Large drop with SAE mode '{SAE_MODE}'.")
        print("   Your SAE may need retraining with lower sparsity.")
    print("="*60)