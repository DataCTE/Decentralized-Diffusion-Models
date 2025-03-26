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
from concurrent.futures import as_completed
from tqdm.auto import tqdm
import torch.distributed as dist
import time
import argparse
import random
import sys

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
            # Add T5 text encoder initialization
            from data.t5 import T5TextEncoder
            self.t5 = T5TextEncoder(self.device, config)
        if 'dino' in enabled_features:
            # Check if we should use existing features or load model
            self.dino = None
            if not getattr(config, 'use_existing_dino', False):
                self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).eval()
        else:
            self.dino = None  # Explicitly set to None if not loading
        
        # Create feature directories aggressively
        self.feature_dir = Path(config.feature_cache_path)
        self._force_create_dirs()

    def _force_create_dirs(self):
        """Create only needed directories"""
        dir_map = {
            'vae': 'latents',
            'clip': 'clip',
            't5': 't5',  # Add directory for T5 embeddings
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

    # Keep only the feature extraction methods that work with single inputs
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

    def _extract_t5_embedding(self, caption):
        """Extract T5 embedding for captions - similar to CLIP but with T5"""
        with torch.no_grad():
            if not isinstance(caption, list):
                caption = [caption]
            
            # Ensure model is on the correct device
            if not hasattr(self.t5.model, 'device') or self.t5.model.device != self.device:
                self.t5.model = self.t5.model.to(self.device)
            
            # Process on GPU
            embedding = self.t5.encode(caption)
            
            # Return CPU version
            return embedding.cpu()

    def _get_bucket_index(self, width, height):
        """Find best matching bucket using config parameters"""
        aspect = width / height
        config = self.config
        
        # 1. Determine aspect ratio group
        aspect_group = None
        
        # Handle the case where bucket_thresholds might be a SimpleNamespace
        if hasattr(config.bucket_thresholds, 'items'):
            # It's a dictionary
            for group, (min_ratio, max_ratio) in config.bucket_thresholds.items():
                if min_ratio <= aspect <= max_ratio:
                    aspect_group = group
                    break
        else:
            # It's a SimpleNamespace - access attributes directly
            if hasattr(config.bucket_thresholds, 'square'):
                min_ratio, max_ratio = config.bucket_thresholds.square
                if min_ratio <= aspect <= max_ratio:
                    aspect_group = 'square'
            
            if aspect_group is None and hasattr(config.bucket_thresholds, 'portrait'):
                min_ratio, max_ratio = config.bucket_thresholds.portrait
                if min_ratio <= aspect <= max_ratio:
                    aspect_group = 'portrait'
            
            if aspect_group is None and hasattr(config.bucket_thresholds, 'landscape'):
                min_ratio, max_ratio = config.bucket_thresholds.landscape
                if min_ratio <= aspect <= max_ratio:
                    aspect_group = 'landscape'
        
        # If no aspect group was determined, use the first bucket
        if aspect_group is None:
            return 0
        
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

    # Update the FEATURE_PROCESSORS dictionary to clearly indicate feature type and save key
    FEATURE_PROCESSORS = {
        'vae': {
            'save_prefix': 'latents',
        },
        'clip': {
            'save_prefix': 'clip',
        },
        't5': {
            'save_prefix': 't5',
        },
        'dino': {
            'save_prefix': 'dino_features',
        },
        'buckets': {
            'save_prefix': 'buckets',
        },
        'dims': {
            'save_prefix': 'dims',
        }
    }

    def run_clustering(self):
        """Pure CPU clustering implementation - standalone process"""
        try:
            # Only rank 0 handles clustering
            if self.rank != 0:
                return True  # Non-rank 0 processes exit immediately
            
            print("Starting CPU-only clustering on rank 0")
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
            # Store file paths to maintain name association
            file_paths = []
            
            with tqdm(total=total_files, desc="Loading features") as pbar:
                for batch_idx in range(0, total_files, batch_size):
                    batch_files = feature_files[batch_idx:batch_idx+batch_size]
                    
                    with ThreadPoolExecutor(max_workers=32) as executor:
                        futures = {executor.submit(self._safe_load_feature, f): (i, f) 
                                 for i, f in enumerate(batch_files, batch_idx)}
                        for future in as_completed(futures):
                            idx, file_path = futures[future]
                            try:
                                feat = future.result()
                                if feat is not None:
                                    mmap_array[idx] = feat.numpy().squeeze()
                                    file_paths.append(file_path)
                            except Exception as e:
                                print(f"Skipping corrupted file: {str(e)}")
                            pbar.update(1)
            
            # Filter invalid entries
            valid_mask = ~np.all(mmap_array == 0, axis=1)
            full_features = mmap_array[valid_mask]
            valid_file_paths = [fp for i, fp in enumerate(file_paths) if valid_mask[i]]
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
            
            # Save cluster labels with original file names
            cluster_dict = {}
            for i, file_path in enumerate(valid_file_paths):
                # Extract base name from file path without rank suffix
                base_name = Path(file_path).stem
                # Remove rank suffix if present (e.g., "_rank0")
                if "_rank" in base_name:
                    base_name = base_name.split("_rank")[0]
                cluster_dict[base_name] = int(cluster_labels[i])
            
            torch.save(cluster_dict, self.feature_dir/"clusters/final_clusters.pt")

            # Cleanup
            os.remove(mmap_path)

            # Save completion marker
            completion_file = self.feature_dir/"clusters/COMPLETED.flag"
            with open(completion_file, 'w') as f:
                f.write(str(time.time()))
            
            return True
            
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            return False

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

    # Optimize dimension and bucket extraction methods
    def _extract_dims_fast(self, img_path):
        """Extract image dimensions without fully loading the image"""
        # Use PIL's faster size property - doesn't require full image loading
        with Image.open(img_path) as img:
            size = img.size  # Just get width and height
        return torch.tensor(size, dtype=torch.int16)

    def _get_bucket_index_from_path(self, img_path):
        """Calculate bucket index directly from image path without full loading"""
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Now calculate bucket index using the dimensions
        return self._get_bucket_index(width, height)

def process_dims_and_buckets(img_path, dims_save_path, buckets_save_path, bucket_func):
    """Process dimensions and buckets together for efficiency."""
    try:
        # Just get the size, don't fully load the image
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Calculate both in one go
        dims_data = torch.tensor((width, height), dtype=torch.int16)
        bucket_idx = bucket_func(width, height)
        bucket_data = torch.tensor(bucket_idx, dtype=torch.int16)
        
        # Save both
        if not dims_save_path.exists():
            torch.save(dims_data, dims_save_path)
        
        if not buckets_save_path.exists():
            torch.save(bucket_data, buckets_save_path)
            
        return True
    except Exception as e:
        print(f"Failed to process {img_path}: {str(e)}")
        return False

def main():
    # Handle standalone clustering without any distributed setup
    if '--clustering' in sys.argv and '--use-existing-dino' in sys.argv:
        config = get_config()
        print("Starting standalone CPU clustering")
        cluster_processor = FeatureGenerator(config, ['clustering'])
        success = cluster_processor.run_clustering()
        sys.exit(0 if success else 1)

    # Parse arguments and determine enabled features
    parser = argparse.ArgumentParser(description='DDM Preprocessing Pipeline')
    parser.add_argument('--buckets', action='store_true', help='Process image buckets')
    parser.add_argument('--clustering', action='store_true', help='Run clustering')
    parser.add_argument('--vae-latents', action='store_true', help='Extract VAE latents')
    parser.add_argument('--clip-latents', action='store_true', help='Extract CLIP embeddings')
    parser.add_argument('--t5-latents', action='store_true', help='Extract T5 embeddings')
    parser.add_argument('--dino-features', action='store_true', help='Extract DINO features')
    parser.add_argument('--all', action='store_true', help='Run all processing stages')
    parser.add_argument('--use-existing-dino', action='store_true',
                     help='Use existing DINO features from disk')
    args = parser.parse_args()

    # Determine enabled features
    enabled_features = []
    if args.all:
        enabled_features = ['vae', 'clip', 't5', 'dino', 'buckets', 'dims', 'clustering']
    else:
        if args.clip_latents:
            enabled_features.append('clip')
        if args.t5_latents:
            enabled_features.append('t5')
        if args.dino_features:
            enabled_features.append('dino')
        if args.vae_latents:
            enabled_features.append('vae')
        if args.buckets:
            enabled_features.append('buckets')
        if args.clustering:
            enabled_features.append('clustering')

    # Initialize distributed only if needed
    needs_distributed = any(feat in ['vae', 'clip', 't5', 'dino'] for feat in enabled_features)
    
    if needs_distributed:
        try:
            rank = int(os.environ['LOCAL_RANK'])
            # ... distributed init ...
        except KeyError:
            print("ERROR: Distributed features require torchrun/MPI")
            sys.exit(1)
    else:
        rank = 0
        world_size = 1

    # Rest of processing code only if not doing standalone clustering
    # ... dataset scanning and processing ...

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        # Force exit with error code
        sys.exit(1) 