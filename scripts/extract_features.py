import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor  # Import from transformers
from config import get_config
from data.dataset import DDMDataset

def extract_features(config_path="config.py", output_dir="/workspace/Decentralized-Diffusion-Models/cache"): # default output dir is now /workspace/.../cache
    """
    Extracts DINOv2 features for the dataset and saves them to disk.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Directory to save the features and dimensions to.
    """
    config = get_config(config_path)
    dataset = DDMDataset(config, split='train') # Initialize dataset to get image paths

    # Load DINOv2 model and processor from Hugging Face Transformers
    model_name = "facebook/dinov2-base"  # You can change to other DINOv2 variants if needed
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda() # Move model to GPU if available
    model.eval() # Set model to evaluation mode

    features_list = []
    dims_list = []

    os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(dataset.image_paths, desc="Extracting features"):
        try:
            image = Image.open(image_path).convert('RGB')
            dims_list.append([image.width, image.height]) # Store image dimensions

            # Preprocess image and extract features
            inputs = processor(images=image, return_tensors="pt").to('cuda') # Move inputs to GPU
            with torch.no_grad():
                outputs = model(**inputs)
                # DINOv2 'base' model returns 'last_hidden_state'
                # You might need to adjust this depending on the DINOv2 variant
                features = outputs.last_hidden_state[:, 0, :] # Take CLS token feature
                features_list.append(features.cpu()) # Move features to CPU to save memory

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue # Skip to the next image in case of error

    # Concatenate all features and dimensions
    all_features = torch.cat(features_list, dim=0)
    all_dims = torch.tensor(dims_list, dtype=torch.int64)

    # Save features and dimensions to disk
    torch.save(all_features, os.path.join(output_dir, "train_features.pt"))
    torch.save(all_dims, os.path.join(output_dir, "dim_cache.pt"))

    print(f"Features saved to {output_dir}")

if __name__ == "__main__":
    extract_features()