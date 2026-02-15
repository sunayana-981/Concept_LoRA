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

# --- IMPORTS FROM YOUR PROJECT ---
# This requires running from the project root so python can find 'tasks' and 'src'
try:
    from tasks.utils import get_sae_and_vit
except ImportError:
    print("CRITICAL ERROR: Could not import 'tasks.utils'.")
    print("Make sure you are running this script from the 'patchsae' root directory.")
    print("Example: python scripts/sae_classification_imported.py")
    sys.exit(1)

# ==========================================
# 1. Configuration
# ==========================================
# Path to your SAE checkpoint (The .pt file)
# If you are using the base SAE from the demo, point to that.
SAE_CHECKPOINT_PATH = "data/sae_weight/base/out.pt" 

DATASET_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/OxfordPets/images"  # Point to your dataset root
LOCAL_CHECKPOINT_PATH = "/home/sunayana/Documents/Concept_LoRA/clip_vitb16_pets_lora_merged.pt"

# Settings for get_sae_and_vit
VIT_TYPE = "base"
BACKBONE = "openai/clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# ==========================================
# 2. Smart Dataset Loader
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
        print(f"   [SmartLoader] Flat Structure: Found {len(self.classes)} classes.")

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
        print(f"   [SmartLoader] Nested Structure (ImageFolder).")
        return dataset
    except:
        sys.stdout = sys.__stdout__
        return FlatFilenameDataset(root=root, transform=transform)

# ==========================================
# 3. Conversion Logic (OpenAI -> HF)
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
# 4. Evaluation with SAE Support
# ==========================================
activation_cache = {}

def get_activation(name):
    def hook(model, input, output):
        activation_cache[name] = output
    return hook

def evaluate_model(model, processor, dataloader, class_names, device, model_name="Model", use_sae=False, sae_model=None):
    print(f"\n--- Evaluating {model_name} (SAE Enabled: {use_sae}) ---")
    model.eval()
    
    hook_handle = None
    if use_sae:
        # Hook the last layer of the Vision Transformer
        target_layer = model.vision_model.encoder.layers[-1] 
        hook_handle = target_layer.register_forward_hook(get_activation('last_layer'))

    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_sae:
                # 1. Run forward to populate hook
                _ = model.get_image_features(images)
                
                # 2. Get activations: [Batch, Seq_Len, Dim]
                internal_feats = activation_cache['last_layer'][0]
                
                # 3. Apply SAE Reconstruction
                if sae_model is not None:
                    b, s, d = internal_feats.shape
                    # Flatten for SAE: [Batch*Seq, Dim]
                    flat_feats = internal_feats.view(-1, d)
                    
                    # === SAE Forward Pass ===
                    sae_output = sae_model(flat_feats)
                    
                    # FIX: Handle tuple return (unpack index 0)
                    if isinstance(sae_output, tuple):
                        reconstructed_flat = sae_output[0]
                    else:
                        reconstructed_flat = sae_output
                    
                    # Reshape back: [Batch, Seq, Dim]
                    internal_feats = reconstructed_flat.view(b, s, d)
                
                # 4. Extract CLS token (index 0)
                cls_token = internal_feats[:, 0, :] 
                
                # 5. Apply Final Layer Norm
                cls_token = model.vision_model.post_layernorm(cls_token)
                
                # 6. Reproject to embedding space
                image_features = model.visual_projection(cls_token)

            else:
                image_features = model.get_image_features(images)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if hook_handle: hook_handle.remove()
    
    accuracy = 100 * correct / total
    print(f"{model_name} Accuracy: {accuracy:.2f}%")
    return accuracy
# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Initializing...")
    
    # 1. Load Dataset
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        dataset = get_smart_dataset(DATASET_ROOT, transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(f"Loaded {len(dataset.classes)} classes.")
    except Exception as e:
        print(f"Dataset Error: {e}")
        sys.exit(1)

    # 2. Load Models using Project Utils
    print("\nLoading SAE and ViT using 'get_sae_and_vit'...")
    try:
        # This loads the SAE and the Base ViT
        sae, vit, cfg = get_sae_and_vit(
            sae_path=SAE_CHECKPOINT_PATH,
            vit_type=VIT_TYPE,
            device=DEVICE,
            backbone=BACKBONE,
            model_path=None,
            classnames=None
        )
        print("SAE and Base ViT loaded successfully.")
        
        # 'vit' here is likely a wrapper, but 'vit.model' is the HF CLIPModel
        # We can use vit.model directly for our evaluation function
        hf_model = vit.model 
        
    except Exception as e:
        print(f"Error loading SAE/ViT: {e}")
        print("Ensure 'tasks.utils' is accessible and paths are correct.")
        sys.exit(1)

    # 3. Setup Processor
    processor = CLIPProcessor.from_pretrained(BACKBONE)

    # 4. Evaluate BASE Model (Before loading LoRA)
    print("\n--- Phase 1: Base Model Evaluation ---")
    acc_base = evaluate_model(hf_model, processor, dataloader, dataset.classes, DEVICE, "Base Model + SAE", use_sae=True, sae_model=sae)

    # 5. Load LoRA Checkpoint into the Model
    print(f"\nLoading LoRA Checkpoint: {LOCAL_CHECKPOINT_PATH}")
    try:
        openai_state_dict = torch.load(LOCAL_CHECKPOINT_PATH, map_location=DEVICE)
        hf_state_dict = convert_openai_to_hf_clip(openai_state_dict)
        hf_model.load_state_dict(hf_state_dict, strict=False)
        print("LoRA Weights merged successfully.")
    except Exception as e:
        print(f"Failed to load LoRA checkpoint: {e}")
        sys.exit(1)

    # 6. Evaluate LoRA Model (With SAE)
    print("\n--- Phase 2: LoRA + SAE Evaluation ---")
    # We pass the 'sae' object returned by get_sae_and_vit
    acc_lora = evaluate_model(hf_model, processor, dataloader, dataset.classes, DEVICE, "LoRA + SAE Model", use_sae=True, sae_model=sae)
    
    print("\n================ Results ================")
    print(f"Base Model Accuracy: {acc_base:.2f}%")
    print(f"LoRA + SAE Accuracy: {acc_lora:.2f}%")