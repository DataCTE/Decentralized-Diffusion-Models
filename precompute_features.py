"""
Centralized preprocessing pipeline for Decentralized Diffusion Models
Combines feature extraction, clustering, and latent precomputation
"""
import os
import uuid
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from torchvision import transforms
from config import get_config
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from data.t5 import T5TextEncoder
from concurrent.futures import ThreadPoolExecutor
import hashlib
from concurrent.futures import as_completed
from tqdm.auto import tqdm
import torch.distributed as dist
import time
import argparse
import pickle
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    def __init__(self, config, enabled_features):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')
        self.enabled_features = enabled_features
        
        # Only initialize requested models
        if 'vae' in enabled_features:
            self.vae = VAEWrapper(self.device, config)
        if 'clip' in enabled_features:
            self.clip = CLIPTextEncoder(self.device, config)
        if 't5' in enabled_features:
            self.t5 = T5TextEncoder(self.device, config)
        if 'dino' in enabled_features and not config.use_existing_dino:
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).eval()
        else:
            self.dino = None  # Explicitly set to None if not loading
        
        # Create feature directories aggressively
        self.feature_dir = Path(config.feature_cache_path)
        self._force_create_dirs()
        
        # Add batch processing buffers
        self.batch_size = 32  # Images per batch
        self.image_buffer = []
        self.caption_buffer = []

    def _force_create_dirs(self):
        """Create only needed directories"""
        dir_map = {
            'vae': 'latents',
            'clip': 'clip',
            't5': 't5',
            'dino': 'dino_features',
            'buckets': 'buckets',
            'dims': 'dims',
            'clustering': 'clusters'
        }
        
        if self.rank == 0:
            for feat in self.enabled_features:
                if feat in dir_map:
                    dir_path = self.feature_dir/dir_map[feat]
                    dir_path.mkdir(parents=True, exist_ok=True)
        dist.barrier()

    def process_image(self, img_path):
        """Batch image processing"""
        self.image_buffer.append(img_path)
        if len(self.image_buffer) >= self.batch_size:
            self._process_batch()
            
    def _process_batch(self):
        """Parallel batch feature extraction"""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._single_process, path) 
                      for path in self.image_buffer]
            for future in as_completed(futures):
                future.result()  # Handle exceptions here
        self.image_buffer.clear()

    def _single_process(self, img_path):
        """Actual processing moved here"""
        try:
            # Check for caption file first
            caption_path = Path(img_path).with_suffix('.txt')
            if not caption_path.exists():
                return False
            
            # Use file contents instead of path for UUID
            with open(img_path, 'rb') as f, open(caption_path, 'rb') as cf:
                img_hash = hashlib.md5(f.read()).hexdigest()
                text_hash = hashlib.md5(cf.read()).hexdigest()
            
            base_name = uuid.UUID(hashlib.md5((img_hash + text_hash).encode()).hexdigest()).hex

            # Process regardless of existing files
            with Image.open(img_path) as img:
                # Handle corrupt images and various color modes
                img = img.convert('RGB')  # Force RGB conversion for all images
                orig_w, orig_h = img.size
                
                # Calculate nearest bucket
                bucket_idx = self._get_bucket_index(orig_w, orig_h)
                features = {
                    'latent': self._extract_vae_latent(img),
                    'clip': self._extract_clip_embedding(caption_path.read_text()),
                    't5': self._extract_t5_embedding(caption_path.read_text()),
                    'dino': self._extract_dino_features(img) if self.dino is not None else None,
                    'dims': torch.tensor(img.size, dtype=torch.int16),
                    'bucket': torch.tensor(bucket_idx, dtype=torch.int16)
                }
            
            # Save with rank-specific naming
            self._save_features(base_name, features)
            return True
        except Exception as e:
            print(f"Rank {self.rank} failed {img_path}: {str(e)}", flush=True)
            return False

    def _extract_vae_latent(self, img):
        """Direct latent encoding without caching"""
        # Ensure RGB and proper tensor format
        img_rgb = img.convert('RGB')
        img_tensor = transforms.ToTensor()(img_rgb.resize((512, 512))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vae.encode(img_tensor).cpu()

    def _extract_clip_embedding(self, caption):
        """Raw text embedding without validation"""
        with torch.no_grad():
            return self.clip.encode([caption]).cpu()

    def _extract_t5_embedding(self, caption):
        """T5 text embedding extraction"""
        with torch.no_grad():
            return self.t5.encode([caption]).cpu()

    def _extract_dino_features(self, img):
        """Robust DINO feature extraction with model check"""
        if self.dino is None:
            return torch.zeros(1, 1024)  # Return empty features if model not loaded
            
        try:
            # Convert to RGB tensor with 3 channels
            img_rgb = img.convert('RGB')
            prep_img = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Handle grayscale
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img_rgb).unsqueeze(0).to(self.device)
            
            # Validate channel dimensions
            if prep_img.shape[1] != 3:
                raise ValueError(f"Invalid channel count: {prep_img.shape[1]}")
            
            with torch.no_grad():
                return self.dino(prep_img).cpu()
        except Exception as e:
            print(f"DINO extraction failed: {str(e)}")
            return torch.zeros(1, 1024)  # Return zero features for corrupt images

    def _save_features(self, base_name, features):
        """Force-save features with rank ID"""
        for feat, data in features.items():
            if feat in self.enabled_features:
                torch.save(data, self.feature_dir/f"{feat}/{base_name}_rank{self.rank}.pt")

    def run_clustering(self):
        """Pure CPU clustering implementation"""
        try:
            if self.rank == 0:
                feature_files = list((self.feature_dir/"dino_features").glob("*.pt"))
                total_files = len(feature_files)
                
                # Memory map setup
                mmap_path = self.feature_dir/"clusters/features.mmap"
                sample_feat = torch.load(feature_files[0], map_location='cpu')
                feat_dim = sample_feat.shape[1]
                
                # Create memory-mapped array
                with open(mmap_path, 'wb') as f:
                    f.seek(total_files * feat_dim * 4 - 1)
                    f.write(b'\0')
                
                mmap_array = np.memmap(mmap_path, dtype=np.float32, mode='r+', 
                                     shape=(total_files, feat_dim))
                
                # Batched loading with error handling
                batch_size = 8192
                with tqdm(total=total_files, desc="Loading features") as pbar:
                    for batch_idx in range(0, total_files, batch_size):
                        batch_files = feature_files[batch_idx:batch_idx+batch_size]
                        
                        with ThreadPoolExecutor(max_workers=32) as executor:
                            futures = {executor.submit(self._safe_load_feature, f): i 
                                     for i, f in enumerate(batch_files, batch_idx)}
                            for future in as_completed(futures):
                                idx = futures[future]
                                try:
                                    feat = future.result()
                                    if feat is not None:
                                        mmap_array[idx] = feat.numpy().squeeze()
                                except Exception as e:
                                    print(f"Skipping corrupted file: {str(e)}")
                                pbar.update(1)
                
                # Filter invalid entries
                valid_mask = ~np.all(mmap_array == 0, axis=1)
                full_features = mmap_array[valid_mask]
                del mmap_array  # Release memory map

                # CPU-only k-means
                kmeans = faiss.Kmeans(
                    full_features.shape[1], 1024,
                    niter=100, 
                    gpu=False,
                    spherical=True,
                    min_points_per_centroid=100,
                    max_points_per_centroid=10000,
                    nredo=3,
                    verbose=True
                )
                kmeans.train(full_features)
                
                # CPU-based hierarchical clustering
                agg = AgglomerativeClustering(
                    n_clusters=8, 
                    linkage='average',
                    metric='cosine', 
                    compute_full_tree=True
                )
                agg.fit(kmeans.centroids)
                
                # Save final clusters
                _, labels = kmeans.index.search(full_features, 1)
                cluster_labels = agg.labels_[labels.flatten()]
                torch.save(cluster_labels, self.feature_dir/"clusters/final_clusters.pt")

                # Cleanup
                os.remove(mmap_path)

            # Simple barrier since we're CPU-only
            if dist.is_initialized():
                dist.barrier()
            
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            if dist.is_initialized():
                dist.barrier()
            raise

    def _safe_load_feature(self, path):
        """Load features with validation and retries"""
        try:
            # Check file size first
            if path.stat().st_size < 100:
                raise ValueError(f"Corrupted small file: {path}")
            
            # Load with device mapping
            data = torch.load(path, map_location='cpu')
            
            # Validate tensor shape
            if data.shape != (1, 1024):
                raise ValueError(f"Invalid feature shape {data.shape} in {path}")
            
            return data
        except (EOFError, RuntimeError, ValueError) as e:
            print(f"Failed to load {path}: {str(e)}")
            return None

    def _get_bucket_index(self, width, height):
        """Find best matching bucket using config parameters"""
        aspect = width / height
        config = self.config
        
        # 1. Determine aspect ratio group
        aspect_group = None
        for group, (min_ratio, max_ratio) in config.bucket_thresholds.items():
            if min_ratio <= aspect <= max_ratio:
                aspect_group = group
                break
            
        # 2. Calculate scaled dimensions
        scale = config.bucket_scale
        scaled_w = round(width / scale) * scale
        scaled_h = round(height / scale) * scale
        
        # 3. Find matching bucket
        for idx, (bw, bh) in enumerate(config.buckets):
            if aspect_group == 'square' and bw == bh and scaled_w == bw and scaled_h == bh:
                return idx
            elif aspect_group == 'portrait' and bw < bh and scaled_w == bw and scaled_h == bh:
                return idx
            elif aspect_group == 'landscape' and bw > bh and scaled_w == bw and scaled_h == bh:
                return idx
            
        # Fallback to nearest bucket
        distances = [
            (abs(scaled_w - bw) + abs(scaled_h - bh), idx)
            for idx, (bw, bh) in enumerate(config.buckets)
        ]
        return min(distances)[1]

    def _extract_dims(self, img):
        """Extract original image dimensions"""
        return torch.tensor(img.size, dtype=torch.int16)

    def _gpu_health_check(self):
        """Verify GPU is responsive"""
        test_tensor = torch.randn(1024, device=self.device)
        torch.cuda.synchronize()
        # If this hangs, it indicates GPU health issues
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

    # Add this class attribute
    FEATURE_PROCESSORS = {
        'vae': {
            'handler': '_extract_vae_latent',
            'save_prefix': 'latents'
        },
        'clip': {
            'handler': '_extract_clip_embedding',
            'save_prefix': 'clip'
        },
        't5': {
            'handler': '_extract_t5_embedding',
            'save_prefix': 't5'
        },
        'dino': {
            'handler': '_extract_dino_features', 
            'save_prefix': 'dino_features'
        },
        'buckets': {
            'handler': '_get_bucket_index',
            'save_prefix': 'buckets'
        },
        'dims': {
            'handler': '_extract_dims',
            'save_prefix': 'dims'
        }
    }

    def _process_image_caption_pair(self, img_path, caption_path):
        """Process a single image-caption pair efficiently"""
        try:
            # Create a unique identifier based on content hashes
            with open(img_path, 'rb') as f, open(caption_path, 'rb') as cf:
                img_hash = hashlib.md5(f.read()).hexdigest()
                text_hash = hashlib.md5(cf.read()).hexdigest()
            
            base_name = uuid.UUID(hashlib.md5((img_hash + text_hash).encode()).hexdigest()).hex
            
            # Read caption text only once
            caption_text = Path(caption_path).read_text()
            
            # Load and process image
            with Image.open(img_path) as img:
                # Force RGB conversion
                img = img.convert('RGB')
                orig_w, orig_h = img.size
                
                # Calculate bucket index
                bucket_idx = self._get_bucket_index(orig_w, orig_h)
                
                # Extract features based on enabled features
                features = {}
                
                # Only process required features
                if 'latent' in self.enabled_features:
                    features['latent'] = self._extract_vae_latent(img)
                
                if 'clip' in self.enabled_features:
                    features['clip'] = self._extract_clip_embedding(caption_text)
                
                if 't5' in self.enabled_features:
                    features['t5'] = self._extract_t5_embedding(caption_text)
                
                if 'dino' in self.enabled_features:
                    features['dino'] = self._extract_dino_features(img)
                
                if 'dims' in self.enabled_features:
                    features['dims'] = torch.tensor(img.size, dtype=torch.int16)
                
                if 'buckets' in self.enabled_features:
                    features['bucket'] = torch.tensor(bucket_idx, dtype=torch.int16)
            
            # Save features
            self._save_features(base_name, features)
            return True
        
        except Exception as e:
            print(f"Rank {self.rank} failed processing {img_path}: {str(e)}")
            return False

def _process_directory(config, rank=0, world_size=1):
    """Efficiently scan and divide dataset for distributed processing"""
    dataset_path = Path(config.dataset_path)
    
    # Only have rank 0 scan the dataset once
    if rank == 0:
        logger.info(f"Scanning dataset directory: {dataset_path}")
        # Use a more efficient file system walk with extension filtering
        image_paths = []
        caption_paths = []
        
        # Cache the lower-cased valid extensions for faster checking
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        
        # Walk the directory tree efficiently using os.walk
        for root, _, files in os.walk(dataset_path):
            for file in files:
                lower_file = file.lower()
                # Check image extensions first
                if any(lower_file.endswith(ext) for ext in valid_exts):
                    img_path = Path(root) / file
                    # Check for caption file with same name but .txt extension
                    caption_path = img_path.with_suffix('.txt')
                    if caption_path.exists():
                        image_paths.append(str(img_path))
                        caption_paths.append(str(caption_path))
        
        print(f"Found {len(image_paths)} valid image-caption pairs")
        
        # Save the paths to a temporary file for other ranks to read
        with open(f"{config.feature_cache_path}/image_paths.pkl", 'wb') as f:
            pickle.dump((image_paths, caption_paths), f)
    
    # Synchronize to ensure rank 0 has finished writing the file
    if world_size > 1:
        dist.barrier()
    
    # All ranks read the paths file
    with open(f"{config.feature_cache_path}/image_paths.pkl", 'rb') as f:
        image_paths, caption_paths = pickle.load(f)
    
    # Distribute paths among ranks
    total_pairs = len(image_paths)
    pairs_per_rank = total_pairs // world_size
    remainder = total_pairs % world_size
    
    # Calculate this rank's start and end indices with balanced distribution
    start_idx = rank * pairs_per_rank + min(rank, remainder)
    end_idx = start_idx + pairs_per_rank + (1 if rank < remainder else 0)
    
    # Get this rank's subset of paths
    rank_image_paths = image_paths[start_idx:end_idx]
    rank_caption_paths = caption_paths[start_idx:end_idx]
    
    print(f"Rank {rank} processing {len(rank_image_paths)} image-caption pairs")
    
    # Clean up the paths file if we're the last rank
    if rank == world_size - 1 and os.path.exists(f"{config.feature_cache_path}/image_paths.pkl"):
        os.remove(f"{config.feature_cache_path}/image_paths.pkl")
    
    return rank_image_paths, rank_caption_paths

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDM Preprocessing Pipeline')
    parser.add_argument('--buckets', action='store_true', help='Process image buckets')
    parser.add_argument('--clustering', action='store_true', help='Run clustering')
    parser.add_argument('--vae-latents', action='store_true', help='Extract VAE latents')
    parser.add_argument('--clip-latents', action='store_true', help='Extract CLIP embeddings')
    parser.add_argument('--t5-embeddings', action='store_true', help='Extract T5 embeddings')
    parser.add_argument('--dino-features', action='store_true', help='Extract DINO features')
    parser.add_argument('--all', action='store_true', help='Run all processing stages')
    parser.add_argument('--use-existing-dino', action='store_true',
                        help='Use existing DINO features from disk')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for I/O operations')
    args = parser.parse_args()

    # Import necessary libraries
    import pickle
    import logging
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Determine enabled features
    enabled_features = []
    if args.all:
        enabled_features = ['vae', 'clip', 't5', 'dino', 'buckets', 'dims', 'clustering']
    else:
        feature_map = {
            'vae-latents': 'vae',
            'clip-latents': 'clip', 
            't5-embeddings': 't5',
            'dino-features': 'dino',
            'buckets': 'buckets',
            'dims': 'dims',
            'clustering': 'clustering'
        }
        enabled_features = [feature_map[f] for f in vars(args) if vars(args)[f] and f in feature_map]

    # Initialize distributed processing
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    logger.info(f"Rank {rank} starting initialization")
    
    # Set device BEFORE initializing process group
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize process group with default settings
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    # Load config after distributed init
    config = get_config()
    
    # Update batch size from args
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Verify dataset path exists
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path {config.dataset_path} not found")
    
    # Create feature cache directory if it doesn't exist
    os.makedirs(config.feature_cache_path, exist_ok=True)
    
    # Get image-caption pairs efficiently
    image_paths, caption_paths = _process_directory(config, rank, world_size)
    
    # Initialize feature generator
    processor = FeatureGenerator(config, enabled_features)
    processor.batch_size = batch_size  # Set batch size from command line
    
    try:
        # Process image-text pairs in batches for better memory efficiency
        total_samples = len(image_paths)
        batch_indices = list(range(0, total_samples, batch_size))
        
        with tqdm(total=total_samples, desc=f"GPU {rank}", position=rank) as pbar:
            for i in range(0, len(batch_indices)):
                start_idx = batch_indices[i]
                end_idx = start_idx + batch_size if i < len(batch_indices) - 1 else total_samples
                
                # Process a batch of image-caption pairs
                batch_image_paths = image_paths[start_idx:end_idx]
                batch_caption_paths = caption_paths[start_idx:end_idx]
                
                # Process batch in parallel
                try:
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = [executor.submit(processor._process_image_caption_pair, 
                                                  img_path, cap_path) 
                                  for img_path, cap_path in zip(batch_image_paths, batch_caption_paths)]
                        
                        # Process results as they complete
                        for future in as_completed(futures):
                            # Handle or log any errors
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"Error processing pair: {str(e)}")
                
                except Exception as batch_e:
                    logger.error(f"Error processing batch: {str(batch_e)}")
                
                # Update progress
                pbar.update(end_idx - start_idx)
                
        # Add GPU health check
        processor._gpu_health_check()
        
        # Clustering phase if enabled
        if 'clustering' in enabled_features:
            # Only rank 0 performs clustering
            if rank == 0:
                logger.info("Starting clustering process...")
                processor.run_clustering()
            
            # Wait for clustering to complete
            if world_size > 1:
                dist.barrier()
            
            logger.info(f"Rank {rank} completed all processing")
    
    except Exception as e:
        logger.error(f"Rank {rank} failed: {str(e)}")
        # Emergency barrier with timeout
        if world_size > 1:
            try:
                dist.barrier(timeout=60)
            except:
                pass
        raise
    finally:
        # Clean up
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 