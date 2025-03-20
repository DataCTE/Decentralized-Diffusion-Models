import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config
# from data.dataset import DDMDataset # No longer need to instantiate the full dataset - REMOVED
import glob # For simpler file discovery
import torch.distributed as dist
from utils.distributed import setup_distributed, get_rank, get_world_size, is_main_process

def extract_features(config_path="config.py", output_dir="/workspace/Decentralized-Diffusion-Models/cache"):
    """
    Extracts DINOv2 features for the dataset in parallel using multiple GPUs and saves them to disk.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Directory to save the features and dimensions to.
    """
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if is_main_process():
        print(f"Using {world_size} GPUs for feature extraction.")

    config = get_config(config_path)
    # dataset = DDMDataset(config, split='train') # Initialize dataset to get image paths - COMPLETELY REMOVED

    # Directly get image paths from the dataset directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    dataset_path = config.dataset_path # Use dataset_path from config
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_path, '**', f'*{ext}'), recursive=True)) # Find all images recursively

    if not image_paths:
        raise FileNotFoundError(f"No images found in dataset path: {dataset_path}. Please check your dataset path in config.py")
    print(f"Found {len(image_paths)} images in dataset path: {dataset_path}")


    # Partition dataset among GPUs
    partition_size = len(image_paths) // world_size
    start_index = rank * partition_size
    end_index = start_index + partition_size
    if rank == world_size - 1:
        end_index = len(image_paths)
    partitioned_image_paths = image_paths[start_index:end_index]

    # Load DINOv2 model and processor
    model_name = "facebook/dinov2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    features_list = []
    dims_list = []

    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    for image_path in tqdm(partitioned_image_paths, desc=f"Rank {rank} Extracting features", position=rank, total=len(partitioned_image_paths)):
        try:
            image = Image.open(image_path).convert('RGB')
            dims_list.append([image.width, image.height])

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features_list.append(features.cpu())

        except Exception as e:
            print(f"Rank {rank} Error processing image {image_path}: {e}")
            continue

    # Gather features and dims from all ranks on rank 0
    all_features_list = [None] * world_size
    all_dims_list = [None] * world_size
    features_tensor = torch.cat(features_list, dim=0) if features_list else torch.empty((0, model.config.hidden_size))
    dims_tensor = torch.tensor(dims_list, dtype=torch.int64) if dims_list else torch.empty((0, 2), dtype=torch.int64)

    dist.gather_object(features_tensor, all_features_list if rank == 0 else None, dst=0)
    dist.gather_object(dims_tensor, all_dims_list if rank == 0 else None, dst=0)

    if is_main_process():
        # Concatenate all features and dimensions on rank 0
        all_features = torch.cat(list(filter(lambda x: x is not None and x.numel() > 0, all_features_list)))
        all_dims = torch.cat(list(filter(lambda x: x is not None and x.numel() > 0, all_dims_list)))

        # Save features and dimensions to disk
        torch.save(all_features, os.path.join(output_dir, "train_features.pt"))
        torch.save(all_dims, os.path.join(output_dir, "dim_cache.pt"))

        print(f"Features saved to {output_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    extract_features()