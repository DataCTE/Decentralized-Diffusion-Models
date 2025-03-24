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

class FeatureGenerator:
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # Initialize models directly without safety checks
        self.vae = VAEWrapper(self.device, config)
        self.clip = CLIPTextEncoder(self.device, config)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).eval()
        
        # Create feature directories aggressively
        self.feature_dir = Path(config.feature_cache_path)
        self._force_create_dirs()
        
        # Add batch processing buffers
        self.batch_size = 8  # Images per batch
        self.image_buffer = []
        self.caption_buffer = []

    def _force_create_dirs(self):
        """Safer directory creation with existence checks"""
        dirs = ['latents', 'clip', 'clusters', 'dims', 'dino_features', 'buckets']
        
        # Only have rank 0 handle directory creation
        if self.rank == 0:
            for d in dirs:
                dir_path = self.feature_dir/d
                if dir_path.exists():
                    shutil.rmtree(dir_path, ignore_errors=True)
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Wait for rank 0 to finish setup
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
                    'dino': self._extract_dino_features(img),
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
        """Robust DINO feature extraction with channel validation"""
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
        torch.save(features['latent'], self.feature_dir/f"latents/{base_name}_rank{self.rank}.pt")
        torch.save(features['clip'], self.feature_dir/f"clip/{base_name}_rank{self.rank}.pt")
        torch.save(features['dino'], self.feature_dir/f"dino_features/{base_name}_rank{self.rank}.pt")
        torch.save(features['dims'], self.feature_dir/f"dims/{base_name}_rank{self.rank}.pt")
        torch.save(features['bucket'], self.feature_dir/f"buckets/{base_name}_rank{self.rank}.pt")

    def run_clustering(self):
        """Global clustering after feature collection"""
        # Wait for all ranks to finish writing
        dist.barrier()
        
        # Load all features across ranks with error handling
        features = []
        feature_files = list((self.feature_dir/"dino_features").glob("*.pt"))
        
        with tqdm(total=len(feature_files), desc="Loading features") as pbar:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(self._safe_load_feature, f): f for f in feature_files}
                for future in as_completed(futures):
                    try:
                        feat = future.result()
                        if feat is not None:
                            features.append(feat)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Skipping corrupted file: {str(e)}")

        if not features:
            raise RuntimeError("No valid features found for clustering")
        
        full_features = torch.cat(features).numpy()
        
        # Paper's exact clustering parameters
        kmeans = faiss.Kmeans(
            full_features.shape[1], 1024,
            niter=100, gpu=True, spherical=True,
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
            nredo=3
        )
        kmeans.train(full_features)
        
        agg = AgglomerativeClustering(
            n_clusters=8, linkage='average',
            metric='cosine', compute_full_tree=True
        )
        agg.fit(kmeans.centroids)
        
        # Save final clusters
        _, labels = kmeans.index.search(full_features, 1)
        cluster_labels = agg.labels_[labels.flatten()]
        torch.save(cluster_labels, self.feature_dir/"clusters/final_clusters.pt")

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

def main():
    # Initialize distributed processing with explicit device settings
    rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank {rank} starting initialization")
    
    # Critical NCCL environment variables
    os.environ["NCCL_ALGO"] = "RING"  # More reliable for small messages
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_MIN_NCHANNELS"] = "12"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_TIMEOUT"] = "300000"  # 5 minute timeout
    
    # Set device BEFORE initializing process group
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize process group with TCP store
    print(f"Rank {rank} initializing process group")
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',  # Explicit TCP initialization
        world_size=int(os.environ['WORLD_SIZE']),
        rank=rank,
        timeout=timedelta(minutes=10)  # Increased timeout
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
    processor = FeatureGenerator(config)
    
    try:
        # Main processing loop
        with tqdm(total=len(local_images), desc=f"GPU {rank}", position=rank) as pbar:
            for img_path in local_images:
                processor.process_image(img_path)
                pbar.update(1)

        # Add final completion marker per rank
        torch.save({'status': 'done'}, processor.feature_dir/f"status_rank{rank}.pt")
        
        # Wait for all ranks to finish processing
        if dist.get_world_size() > 1:
            dist.barrier()

        # Cluster only after all ranks confirm completion
        if rank == 0:
            try:
                # Add timeout for status check
                start_time = time.time()
                while not all((processor.feature_dir/f"status_rank{r}.pt").exists() 
                            for r in range(dist.get_world_size())):
                    if time.time() - start_time > 300:  # 5 minute timeout
                        raise TimeoutError("Not all ranks completed within 5 minutes")
                    time.sleep(1)
                
                processor.run_clustering()
            except Exception as e:
                print(f"Clustering failed: {str(e)}")
        else:
            # Non-zero ranks wait for final signal
            while not (processor.feature_dir/"clusters/final_clusters.pt").exists():
                time.sleep(1)

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