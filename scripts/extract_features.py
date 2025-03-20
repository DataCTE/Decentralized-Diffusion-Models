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
    Extracts DINOv2 features with single-threaded CPU file discovery (rank 0 only) and GPU feature extraction.
    File discovery is CPU-bound and performed on CPU (single-threaded) on rank 0 only.
    Discovered file paths are broadcast to all ranks.
    Feature extraction is GPU-accelerated.
    """
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        # Initialize device first
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        
        # Initialize process group with explicit device_id
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            device_id=local_rank
        )
        
        world_size = dist.get_world_size()
        print(f"Rank {rank}: Using device: {device}")
        if is_main_process():
            print(f"Using {world_size} GPUs for feature extraction.")
    else: # Single GPU or CPU mode
        rank, world_size = 0, 1 # Set rank and world_size for single process
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use CUDA if available, else CPU
        torch.cuda.set_device(device) if torch.cuda.is_available() else None # Set device if CUDA is available
        print(f"Using device: {device}")


    config = get_config(config_path)

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    dataset_path = config.dataset_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}. Please check your config.py")

    if is_main_process(): # File discovery only on main process (rank 0)
        print(f"Rank {rank}: Starting file discovery in {dataset_path} (CPU-bound, single-threaded)")
        discovery_start_time = time.time()
        
        # Initialize progress bar with rate display
        with tqdm(unit='img', unit_scale=True, desc=f"Rank {rank} Discovering files") as progress_bar:
            for ext in image_extensions:
                pattern = os.path.join(dataset_path, f'*{ext}')
                extension_image_paths = glob.glob(pattern)
                image_paths.extend(extension_image_paths)
                progress_bar.update(len(extension_image_paths))  # Update with number of images found

        discovery_duration = time.time() - discovery_start_time
        print(f"\nRank {rank}: File discovery completed in {discovery_duration:.2f} seconds. Found {len(image_paths)} images.")

    else: # Non-main processes (ranks > 0)
        image_paths = [None] * world_size # Initialize image_paths list to receive broadcast

    if world_size > 1:
        print(f"Rank {rank}: Waiting to receive image paths...")
        dist.barrier()  # Add barrier to ensure all processes wait for file discovery
        dist.broadcast_object_list(object_list=[image_paths], src=0) # Broadcast from rank 0 to all ranks
        image_paths = image_paths[0] # Extract received image_paths
        print(f"Rank {rank}: Received image paths. Total images: {len(image_paths)}")


    if not image_paths:
        raise FileNotFoundError(f"No images found in dataset path: {dataset_path}. Please check your dataset path in config.py and image formats.")


    # Partition dataset among GPUs (rest of the script remains largely the same)
    partition_size = len(image_paths) // world_size
    start_index = rank * partition_size
    end_index = start_index + partition_size
    if world_size > 1: # Only partition if using multiple processes
        if rank == world_size - 1:
            end_index = len(image_paths)
        partitioned_image_paths = image_paths[start_index:end_index]
    else: # Single process - use all image paths
        partitioned_image_paths = image_paths


    model_name = "facebook/dinov2-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Rank {rank}: Model device: {next(model.parameters()).device}")

    features_dir = os.path.join(output_dir, "features")
    dims_dir = os.path.join(output_dir, "dimensions")

    if is_main_process(): # Only main process in distributed setup or single process
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


    if is_main_process(): # Only main process in distributed setup or single process
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\nTotal feature extraction time: {total_duration:.2f} seconds")
        print(f"Features saved to {features_dir}")
        print(f"Dimensions saved to {dims_dir}")
        if world_size > 1:
            dist.destroy_process_group() # Destroy process group only in distributed mode


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