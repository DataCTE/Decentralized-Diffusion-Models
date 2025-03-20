import os
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get_config
from utils.distributed import setup_distributed, get_rank, get_world_size, is_main_process
import glob

def extract_features(config_path="config.py", output_dir="cache"):
    """Main feature extraction workflow with distributed support"""
    # Distributed setup
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank, world_size, device = distributed_setup()
    else:
        rank, world_size, device = single_process_setup()

    config = get_config(config_path)
    validate_dataset_path(config.dataset_path)

    # File discovery and distribution
    image_paths = handle_file_discovery(config.dataset_path, rank, world_size)
    
    # Feature extraction pipeline
    process_images(image_paths, device, output_dir, rank, world_size)

def distributed_setup():
    """Configure distributed training environment"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://'
    )
    
    world_size = dist.get_world_size()
    print(f"Rank {rank}: Using device: {device}")
    return rank, world_size, device

def single_process_setup():
    """Configure single process environment"""
    rank, world_size = 0, 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"Using device: {device}")
    return rank, world_size, device

def validate_dataset_path(dataset_path):
    """Ensure dataset path exists"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

def handle_file_discovery(dataset_path, rank, world_size):
    """Discover and distribute image paths using dataset.py's logic"""
    image_paths = []
    
    if is_main_process():
        print(f"Rank {rank}: Starting file discovery in {dataset_path}")
        image_paths = discover_valid_images(dataset_path)
        
        if not image_paths:
            raise FileNotFoundError(f"No valid images found in {dataset_path}")
            
        print(f"Rank {rank}: Found {len(image_paths)} valid images")
    
    if world_size > 1:
        image_paths = distribute_image_paths(image_paths, rank, world_size)
    
    return image_paths

def discover_valid_images(dataset_path):
    """Mirror dataset.py's file discovery exactly"""
    
    
    # Match dataset.py's discovery pattern exactly
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', f'*{ext}'), recursive=True))
    
    # Match dataset.py's validation parameters
    min_size = 256  # Must match dataset.py's min_size default
    batch_size = 500  # Match dataset.py's processing batch size
    
    valid_files = []
    
    # Mirror dataset.py's processing exactly
    with tqdm(total=len(image_files), desc="Validating images", unit="img") as pbar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_batch, batch, min_size) 
                      for batch in chunked(image_files, batch_size)]
            
            for future in as_completed(futures):
                batch_valid, batch_processed = future.result()
                valid_files.extend(batch_valid)
                pbar.update(batch_processed)
    
    return valid_files

def process_batch(batch, min_size):
    """Batch processor matching dataset.py's logic"""
    valid = []
    processed = 0
    
    for img_path in batch:
        processed += 1
        try:
            with Image.open(img_path) as img:
                # Match dataset.py's exact validation criteria
                width, height = img.size
                if width >= min_size and height >= min_size:
                    valid.append(img_path)
        except Exception:
            continue
            
    return valid, processed

def distribute_image_paths(image_paths, rank, world_size):
    """Handle distributed path distribution"""
    if is_main_process():
        print(f"Rank {rank}: Broadcasting {len(image_paths)} image paths")
        dist.broadcast_object_list([image_paths], src=0)
    else:
        image_paths = [None]
        dist.broadcast_object_list(image_paths, src=0)
        image_paths = image_paths[0]
    
    print(f"Rank {rank}: Received {len(image_paths)} images")
    return partition_data(image_paths, rank, world_size)

def partition_data(data, rank, world_size):
    """Split data evenly across processes"""
    partition_size = len(data) // world_size
    start = rank * partition_size
    end = start + partition_size
    if rank == world_size - 1:
        end = len(data)
    return data[start:end]

def process_images(image_paths, device, output_dir, rank, world_size):
    """Core feature extraction pipeline"""
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dimensions"), exist_ok=True)

    model = load_dino_model(device)
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_image, path, processor, model, device, output_dir)
                   for path in image_paths]
        
        with tqdm(total=len(image_paths), desc=f"Rank {rank} Processing", position=rank+1) as pbar:
            for future in as_completed(futures):
                future.result()  # Handle exceptions here if needed
                pbar.update(1)

def load_dino_model(device):
    """Load and configure DINOv2 model"""
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model

def process_single_image(image_path, processor, model, device, output_dir):
    """Process individual image and save features"""
    try:
        with Image.open(image_path) as img:
            # Extract dimensions first
            dims = torch.tensor(img.size, dtype=torch.int64)
            
            # Process image
            inputs = processor(images=img.convert('RGB'), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].cpu()

            # Save results
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            torch.save(features, os.path.join(output_dir, "features", f"{base_name}.pt"))
            torch.save(dims, os.path.join(output_dir, "dimensions", f"{base_name}.pt"))
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def chunked(lst, n):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    extract_features()
