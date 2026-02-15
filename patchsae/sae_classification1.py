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
# CRITICAL PARAMETER: Which layer to apply SAE
# ==========================================
# Layer 11 (last): Patches don't affect output anymore → No SAE effect
# Layer 6-9 (middle): Patches still flow through attention → SAE has effect
# Layer 0-3 (early): Too early, may not capture meaningful features

SAE_LAYER = 8  # Try 6, 7, 8, or 9 for actual SAE effects

# SAE application mode
SAE_MODE = 'preserve_cls'  # 'preserve_cls', 'blend', 'original'
BLEND_ALPHA = 0.2

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
    """Preserve CLS token, only reconstruct patches."""
    b, s, d = internal_feats.shape
    
    cls_token = internal_feats[:, 0:1, :]
    patch_tokens = internal_feats[:, 1:, :]
    
    patch_flat = patch_tokens.reshape(-1, d)
    
    sae_output = sae_model(patch_flat)
    if isinstance(sae_output, tuple):
        patch_reconstructed = sae_output[0]
    else:
        patch_reconstructed = sae_output
    
    patch_reconstructed = patch_reconstructed.view(b, s-1, d)
    reconstructed = torch.cat([cls_token, patch_reconstructed], dim=1)
    
    return reconstructed

def apply_sae_blended(internal_feats, sae_model, alpha=0.2):
    """Blend original and reconstructed features."""
    b, s, d = internal_feats.shape
    
    flat_feats = internal_feats.view(-1, d)
    sae_output = sae_model(flat_feats)
    if isinstance(sae_output, tuple):
        reconstructed_flat = sae_output[0]
    else:
        reconstructed_flat = sae_output
    
    reconstructed = reconstructed_flat.view(b, s, d)
    blended = alpha * reconstructed + (1.0 - alpha) * internal_feats
    
    return blended

def apply_sae_original(internal_feats, sae_model):
    """Apply SAE to all tokens."""
    b, s, d = internal_feats.shape
    flat_feats = internal_feats.view(-1, d)
    
    sae_output = sae_model(flat_feats)
    if isinstance(sae_output, tuple):
        reconstructed_flat = sae_output[0]
    else:
        reconstructed_flat = sae_output
    
    return reconstructed_flat.view(b, s, d)

# ==========================================
# Custom Forward Hook for Mid-Layer SAE
# ==========================================
class SAEHookHandler:
    def __init__(self, model, sae_model, sae_layer, sae_mode='preserve_cls', blend_alpha=0.2):
        self.model = model
        self.sae_model = sae_model
        self.sae_layer = sae_layer
        self.sae_mode = sae_mode
        self.blend_alpha = blend_alpha
        self.hook_handle = None
        
    def hook_fn(self, module, input, output):
        """Hook function that modifies the output of the target layer."""
        # Output from transformer layer is typically a tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Apply SAE based on mode
        if self.sae_mode == 'preserve_cls':
            modified = apply_sae_preserve_cls(hidden_states, self.sae_model)
        elif self.sae_mode == 'blend':
            modified = apply_sae_blended(hidden_states, self.sae_model, self.blend_alpha)
        else:  # 'original'
            modified = apply_sae_original(hidden_states, self.sae_model)
        
        # Return in same format as input
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        else:
            return modified
    
    def register(self):
        """Register the hook on the specified layer."""
        target_layer = self.model.vision_model.encoder.layers[self.sae_layer]
        self.hook_handle = target_layer.register_forward_hook(self.hook_fn)
        print(f"   Registered SAE hook at layer {self.sae_layer}")
    
    def remove(self):
        """Remove the hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

# ==========================================
# Evaluation Function
# ==========================================
def evaluate_model(model, processor, dataloader, class_names, device, 
                   model_name="Model", use_sae=False, sae_hook=None):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"SAE Enabled: {use_sae}")
    if use_sae and sae_hook:
        print(f"SAE Layer: {sae_hook.sae_layer}")
        print(f"SAE Mode: {sae_hook.sae_mode}")
    print(f"{'='*60}")
    
    model.eval()
    
    # Register hook if using SAE
    if use_sae and sae_hook:
        sae_hook.register()

    # Prepare text features
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass (SAE is applied via hook if registered)
            image_features = model.get_image_features(images)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity and predictions
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Remove hook
    if use_sae and sae_hook:
        sae_hook.remove()
    
    accuracy = 100 * correct / total
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*60}\n")
    
    return accuracy

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("SAE Classification - Mid-Layer Application")
    print(f"{'='*60}")
    print(f"SAE Layer: {SAE_LAYER} (out of 12 layers, 0-11)")
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

    # Create SAE hook handler
    sae_hook = SAEHookHandler(hf_model, sae, SAE_LAYER, SAE_MODE, BLEND_ALPHA)

    # 3. Baseline (No SAE)
    print("\n" + "="*60)
    print("PHASE 1: BASELINE (NO SAE)")
    print("="*60)
    acc_base_no_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        "Base Model (No SAE)", use_sae=False
    )

    # 4. Base Model WITH SAE at mid-layer
    print("\n" + "="*60)
    print(f"PHASE 2: BASE MODEL WITH SAE AT LAYER {SAE_LAYER}")
    print("="*60)
    acc_base_with_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        f"Base Model (SAE-Layer{SAE_LAYER})", use_sae=True, sae_hook=sae_hook
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
    print(f"PHASE 4: LORA MODEL WITH SAE AT LAYER {SAE_LAYER}")
    print("="*60)
    acc_lora_with_sae = evaluate_model(
        hf_model, processor, dataloader, dataset.classes, DEVICE, 
        f"LoRA Model (SAE-Layer{SAE_LAYER})", use_sae=True, sae_hook=sae_hook
    )
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Base (No SAE):         {acc_base_no_sae:.2f}%")
    print(f"Base (SAE-Layer{SAE_LAYER}):    {acc_base_with_sae:.2f}%")
    print(f"LoRA (No SAE):         {acc_lora_no_sae:.2f}%")
    print(f"LoRA (SAE-Layer{SAE_LAYER}):    {acc_lora_with_sae:.2f}%")
    print("="*60)
    
    # Analysis
    base_drop = 100 * (1 - acc_base_with_sae / acc_base_no_sae) if acc_base_no_sae > 0 else 0
    lora_drop = 100 * (1 - acc_lora_with_sae / acc_lora_no_sae) if acc_lora_no_sae > 0 else 0
    
    print(f"\nAccuracy Drop from SAE:")
    print(f"  Base: {base_drop:.1f}%")
    print(f"  LoRA: {lora_drop:.1f}%")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if abs(base_drop) < 1:
        print(f"⚠️  SAE at layer {SAE_LAYER} has NO EFFECT on accuracy!")
        print(f"   Try a DIFFERENT layer (6, 7, 8, or 9)")
        print(f"   OR your SAE may not be trained for this layer")
    elif base_drop < 10:
        print(f"✅ Excellent! SAE at layer {SAE_LAYER} maintains accuracy well.")
    elif base_drop < 25:
        print(f"✓  Good. SAE causes acceptable accuracy drop.")
    else:
        print(f"❌ Large drop. SAE at layer {SAE_LAYER} is too destructive.")
        print(f"   Try 'blend' mode with lower alpha")
    
    print("="*60)