import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
from data.dataset import DDMDataset
import torch.distributed as dist  # Import distributed library
from utils.distributed import setup_distributed, get_rank, get_world_size, is_main_process # Import distributed utils

def extract_features(config_path="config.py", output_dir="/workspace/Decentralized-Diffusion-Models/cache"):
    """
    Extracts DINOv2 features for the dataset in parallel using multiple GPUs and saves them to disk.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Directory to save the features and dimensions to.
    """
    rank, world_size = setup_distributed() # Initialize distributed environment
    device = torch.device(f"cuda:{rank}")

    if is_main_process():
        print(f"Using {world_size} GPUs for feature extraction.")

    config = get_config(config_path)
    dataset = DDMDataset(config, split='train') # Initialize dataset to get image paths

    # Partition dataset among GPUs
    partition_size = len(dataset.image_paths) // world_size
    start_index = rank * partition_size
    end_index = start_index + partition_size
    if rank == world_size - 1: # Last process takes remaining images
        end_index = len(dataset.image_paths)
    partitioned_image_paths = dataset.image_paths[start_index:end_index]

    # Load DINOv2 model and processor
    model_name = "facebook/dinov2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device) # Move model to current GPU
    model.eval()

    features_list = []
    dims_list = []

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(partitioned_image_paths, desc=f"Rank {rank} Extracting features", position=rank, total=len(partitioned_image_paths)): # Add tqdm per rank
        try:
            image = Image.open(image_path).convert('RGB')
            dims_list.append([image.width, image.height])

            inputs = processor(images=image, return_tensors="pt").to(device) # Move inputs to current GPU
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features_list.append(features.cpu()) # Keep features on CPU for gathering

        except Exception as e:
            print(f"Rank {rank} Error processing image {image_path}: {e}")
            continue

    # Gather features and dims from all ranks on rank 0
    all_features_list = [None] * world_size
    all_dims_list = [None] * world_size
    features_tensor = torch.cat(features_list, dim=0) if features_list else torch.empty((0, model.config.hidden_size)) # Handle empty list case
    dims_tensor = torch.tensor(dims_list, dtype=torch.int64) if dims_list else torch.empty((0, 2), dtype=torch.int64) # Handle empty list case

    dist.gather_object(features_tensor, all_features_list if rank == 0 else None, dst=0) # Gather tensors directly
    dist.gather_object(dims_tensor, all_dims_list if rank == 0 else None, dst=0)

    if is_main_process():
        # Concatenate all features and dimensions on rank 0
        all_features = torch.cat(list(filter(lambda x: x is not None and x.numel() > 0, all_features_list))) # Filter out None and empty tensors
        all_dims = torch.cat(list(filter(lambda x: x is not None and x.numel() > 0, all_dims_list))) # Filter out None and empty tensors

        # Save features and dimensions to disk
        torch.save(all_features, os.path.join(output_dir, "train_features.pt"))
        torch.save(all_dims, os.path.join(output_dir, "dim_cache.pt"))

        print(f"Features saved to {output_dir}")

    dist.destroy_process_group() # Clean up distributed processes

if __name__ == "__main__":
    extract_features()