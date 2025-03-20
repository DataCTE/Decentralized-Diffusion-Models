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
import time  # Import the time module

def extract_features(config_path="config.py", output_dir="/workspace/Decentralized-Diffusion-Models/cache"):
    """
    Extracts DINOv2 features for the dataset in parallel using multiple GPUs and saves them to disk as individual files.
    Includes progress bar with average time per image.

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

    # Create separate output directories for features and dimensions
    features_dir = os.path.join(output_dir, "features")
    dims_dir = os.path.join(output_dir, "dimensions")

    if is_main_process():
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(dims_dir, exist_ok=True)

    start_time = time.time() # Start timer for total duration
    image_times = [] # List to store processing times for last 10 images

    for image_path in tqdm(partitioned_image_paths, desc=f"Rank {rank} Extracting features", position=rank, total=len(partitioned_image_paths)):
        image_process_start_time = time.time() # Start timer for image processing
        try:
            image = Image.open(image_path).convert('RGB')
            dims = torch.tensor([image.width, image.height], dtype=torch.int64)

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].cpu() # Move features to CPU immediately

            # Construct base filename from image path (remove extension and path prefix)
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            feature_filename = f"{base_filename}.pt"
            dim_filename = f"{base_filename}.pt" # Use .pt extension for dimensions as well for consistency

            # Save feature and dimension files
            torch.save(features, os.path.join(features_dir, feature_filename))
            torch.save(dims, os.path.join(dims_dir, dim_filename))


        except Exception as e:
            print(f"Rank {rank} Error processing image {image_path}: {e}")
            continue
        finally: # Ensure time is recorded even if there's an exception
            image_process_end_time = time.time()
            image_process_time = image_process_end_time - image_process_start_time
            image_times.append(image_process_time)

            if len(image_times) > 10: # Keep only last 10 times
                image_times.pop(0)

            avg_time_10_images = sum(image_times) / len(image_times) if image_times else 0
            images_per_sec = 1 / avg_time_10_images if avg_time_10_images > 0 else 0

            description = f"Rank {rank} Extracting features - Avg time/image (last 10): {avg_time_10_images:.3f}s, Images/sec: {images_per_sec:.2f}"
            tqdm.write("\r" + description, end='') # Update tqdm description


    if is_main_process():
        end_time = time.time() # End timer for total duration
        total_duration = end_time - start_time
        print(f"\nTotal feature extraction time: {total_duration:.2f} seconds")
        print(f"Features saved to {features_dir}")
        print(f"Dimensions saved to {dims_dir}")

    dist.destroy_process_group()

if __name__ == "__main__":
    extract_features()