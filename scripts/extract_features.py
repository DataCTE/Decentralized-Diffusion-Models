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
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for multithreading

def extract_features(config_path="config.py", output_dir="/workspace/Decentralized-Diffusion-Models/cache"):
    """
    Extracts DINOv2 features with multithreaded file discovery and processing for datasets
    where all images are in a single directory.
    File discovery is CPU-bound and performed on CPU. Feature extraction is GPU-accelerated.
    """
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Rank {rank}: Using device: {device}")

    if is_main_process():
        print(f"Using {world_size} GPUs for feature extraction.")

    config = get_config(config_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    dataset_path = config.dataset_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}. Please check your config.py")

    print(f"Rank {rank}: Starting file discovery in {dataset_path} (CPU-bound)")
    discovery_start_time = time.time()
    image_discovery_times = [] # To store times for image discovery iterations
    processed_iterations_count = 0


    # File discovery - CPU-bound operations (glob.glob) - performed on CPU
    with ThreadPoolExecutor(max_workers=8) as executor: # Adjust max_workers as needed
        futures = []
        for ext in image_extensions:
            pattern = os.path.join(dataset_path, f'*{ext}') # Direct glob in dataset_path
            future = executor.submit(glob.glob, pattern) # Run glob.glob in thread
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(image_extensions), desc=f"Rank {rank} Discovering files"): # Progress over image extensions
            iteration_start_time = time.time()
            image_paths.extend(future.result()) # Collect image paths from each thread
            iteration_duration = time.time() - iteration_start_time
            image_discovery_times.append(iteration_duration)
            processed_iterations_count += 1

            if len(image_discovery_times) > 10:
                image_discovery_times.pop(0) # Keep only last 10 times

            avg_time_10_iterations = sum(image_discovery_times) / len(image_discovery_times) if image_discovery_times else 0
            iterations_per_sec = 1 / avg_time_10_iterations if avg_time_10_iterations > 0 else 0

            description = f"Rank {rank} Discovering files - Avg time/iteration (last 10): {avg_time_10_iterations:.3f}s, Iterations/sec: {iterations_per_sec:.2f}"
            tqdm.write("\r" + description, end='')


    discovery_duration = time.time() - discovery_start_time
    print(f"\nRank {rank}: File discovery completed in {discovery_duration:.2f} seconds. Found {len(image_paths)} images.")


    if not image_paths:
        raise FileNotFoundError(f"No images found in dataset path: {dataset_path}. Please check your dataset path in config.py and image formats.")


    # Partition dataset among GPUs (rest of the script remains largely the same)
    partition_size = len(image_paths) // world_size
    start_index = rank * partition_size
    end_index = start_index + partition_size
    if rank == world_size - 1:
        end_index = len(image_paths)
    partitioned_image_paths = image_paths[start_index:end_index]

    model_name = "facebook/dinov2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Rank {rank}: Model device: {next(model.parameters()).device}")

    features_dir = os.path.join(output_dir, "features")
    dims_dir = os.path.join(output_dir, "dimensions")

    if is_main_process():
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(dims_dir, exist_ok=True)

    start_time = time.time()
    image_times = []
    processed_images_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for image_path in partitioned_image_paths:
            future = executor.submit(
                process_single_image,
                image_path,
                processor,
                model,
                device,
                features_dir,
                dims_dir
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(partitioned_image_paths), desc=f"Rank {rank} Extracting features", position=rank):
            image_process_time = future.result()
            image_times.append(image_process_time)
            processed_images_count += 1

            if len(image_times) > 10:
                image_times.pop(0)

            avg_time_10_images = sum(image_times) / len(image_times) if image_times else 0
            images_per_sec = 1 / avg_time_10_images if avg_time_10_images > 0 else 0

            description = f"Rank {rank} Extracting features - Avg time/image (last 10): {avg_time_10_images:.3f}s, Images/sec: {images_per_sec:.2f}"
            tqdm.write("\r" + description, end='')


    if is_main_process():
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\nTotal feature extraction time: {total_duration:.2f} seconds")
        print(f"Features saved to {features_dir}")
        print(f"Dimensions saved to {dims_dir}")

    dist.destroy_process_group()


def process_single_image(image_path, processor, model, device, features_dir, dims_dir):
    image_process_start_time = time.time()
    try:
        image = Image.open(image_path).convert('RGB')
        dims = torch.tensor([image.width, image.height], dtype=torch.int64).to(device)

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        feature_filename = f"{base_filename}.pt"
        dim_filename = f"{base_filename}.pt"

        torch.save(features.cpu(), os.path.join(features_dir, feature_filename))
        torch.save(dims.cpu(), os.path.join(dims_dir, dim_filename))

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    finally:
        image_process_end_time = time.time()
        image_process_time = image_process_end_time - image_process_start_time
        return image_process_time


if __name__ == "__main__":
    extract_features()