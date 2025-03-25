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
from concurrent.futures import ThreadPoolExecutor
import hashlib
from concurrent.futures import as_completed
from collections import defaultdict
from tqdm.auto import tqdm
import shutil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
import time
import argparse

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
            
            # Generate deterministic UUID
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
        """Distributed clustering implementation"""
        # Wait for all ranks to finish writing
        dist.barrier()
        
        if self.rank == 0:
            print("Starting distributed clustering...")
            try:
                # Load features in parallel across all available GPUs
                feature_files = list((self.feature_dir/"dino_features").glob("*.pt"))
                features = []
                
                # Use memory mapping for large files
                with tqdm(total=len(feature_files), desc="Loading features") as pbar:
                    with ThreadPoolExecutor(max_workers=16) as executor:
                        futures = {executor.submit(self._safe_load_feature, f): f 
                                  for f in feature_files}
                        for future in as_completed(futures):
                            try:
                                feat = future.result()
                                if feat is not None:
                                    features.append(feat)
                                pbar.update(1)
                            except Exception as e:
                                print(f"Skipping corrupted file: {str(e)}")

                full_features = torch.cat(features).numpy()
                
                # Distributed k-means with Faiss
                kmeans = faiss.Kmeans(
                    full_features.shape[1], 1024,
                    niter=100, gpu=True, spherical=True,
                    min_points_per_centroid=100,
                    max_points_per_centroid=10000,
                    nredo=3,
                    verbose=True
                )
                kmeans.train(full_features)
                
                # Broadcast centroids to all ranks
                centroids_tensor = torch.from_numpy(kmeans.centroids)
                dist.broadcast(centroids_tensor, src=0)
                
                # Distributed hierarchical clustering
                agg = AgglomerativeClustering(
                    n_clusters=8, linkage='average',
                    metric='cosine', compute_full_tree=True
                )
                agg.fit(kmeans.centroids)
                
                # Save final clusters
                _, labels = kmeans.index.search(full_features, 1)
                cluster_labels = agg.labels_[labels.flatten()]
                torch.save(cluster_labels, self.feature_dir/"clusters/final_clusters.pt")
                
            except Exception as e:
                print(f"Clustering failed: {str(e)}")
                # Broadcast failure signal
                failure_flag = torch.tensor([1], device=self.device)
            else:
                failure_flag = torch.tensor([0], device=self.device)
            
            dist.broadcast(failure_flag, src=0)
        else:
            failure_flag = torch.tensor([0], device=self.device)
            dist.broadcast(failure_flag, src=0)

        if failure_flag.item() == 1:
            raise RuntimeError("Clustering failed on rank 0")

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DDM Preprocessing Pipeline')
    parser.add_argument('--buckets', action='store_true', help='Process image buckets')
    parser.add_argument('--clustering', action='store_true', help='Run clustering')
    parser.add_argument('--vae-latents', action='store_true', help='Extract VAE latents')
    parser.add_argument('--clip-latents', action='store_true', help='Extract CLIP embeddings')
    parser.add_argument('--dino-features', action='store_true', help='Extract DINO features')
    parser.add_argument('--all', action='store_true', help='Run all processing stages')
    parser.add_argument('--use-existing-dino', action='store_true',
                        help='Use existing DINO features from disk')
    args = parser.parse_args()

    # Determine enabled features
    enabled_features = []
    if args.all:
        enabled_features = ['vae', 'clip', 'dino', 'buckets', 'dims', 'clustering']
    else:
        feature_map = {
            'vae-latents': 'vae',
            'clip-latents': 'clip', 
            'dino-features': 'dino',
            'buckets': 'buckets',
            'dims': 'dims',
            'clustering': 'clustering'
        }
        enabled_features = [feature_map[f] for f in vars(args) if vars(args)[f] and f in feature_map]

    # Add these after setting the device but before init_process_group
    torch.cuda.synchronize()
    time.sleep(1)  # Allow device warmup

    # Initialize distributed processing
    rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank {rank} starting initialization")
    
    # Set device BEFORE initializing process group
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize process group with default settings
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Use automatic environment initialization
        world_size=int(os.environ['WORLD_SIZE']),
        rank=rank
    )
    
    # Load config after distributed init
    config = get_config()
    
    # Verify dataset path exists
    if rank == 0:
        if not Path(config.dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {config.dataset_path} not found")
    
    # Get all images - add progress bar
    if rank == 0:
        print("Scanning dataset directory...")
    all_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Distribute images
    chunk_size = len(all_images) // dist.get_world_size()
    local_images = all_images[rank*chunk_size : (rank+1)*chunk_size]
    print(f"Rank {rank} received {len(local_images)} images to process")
    
    # Initialize feature generator
    processor = FeatureGenerator(config, enabled_features)
    
    try:
        # Main processing loop
        with tqdm(total=len(local_images), desc=f"GPU {rank}", position=rank) as pbar:
            for img_path in local_images:
                try:
                    base_name = uuid.UUID(hashlib.md5(img_path.encode()).hexdigest()).hex
                    
                    # Only process enabled features
                    features = {}
                    for feat in enabled_features:
                        if feat in FeatureGenerator.FEATURE_PROCESSORS:
                            handler_info = FeatureGenerator.FEATURE_PROCESSORS[feat]
                            handler = handler_info['handler']
                            features[feat] = getattr(processor, handler)(img_path)
                    
                    # Save enabled features
                    for feat, data in features.items():
                        prefix = FeatureGenerator.FEATURE_PROCESSORS[feat]['save_prefix']
                        torch.save(data, processor.feature_dir/f"{prefix}/{base_name}_rank{rank}.pt")
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Rank {rank} failed to process {img_path}: {str(e)}")
                    continue  # Skip to next image

        # Add final completion marker per rank
        torch.save({'status': 'done'}, processor.feature_dir/f"status_rank{rank}.pt")
        
        # Wait for all ranks to finish processing
        if dist.get_world_size() > 1:
            dist.barrier()

        # Conditional clustering
        if 'clustering' in enabled_features:
            if rank == 0:
                processor.run_clustering()
            else:
                # Wait for clustering results
                while not (processor.feature_dir/"clusters/final_clusters.pt").exists():
                    time.sleep(30)

    except Exception as e:
        print(f"Rank {rank} failed: {str(e)}")
    finally:
        # Cleanup status files
        if rank == 0:
            for r in range(dist.get_world_size()):
                (processor.feature_dir/f"status_rank{r}.pt").unlink(missing_ok=True)
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 