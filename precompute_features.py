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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return self.image_paths[idx]

class FeatureGenerator:
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f'cuda:{self.rank}')
        
        # Add batch processing parameters
        self.batch_size = 16  # Adjust based on GPU memory
        self.num_workers = 8  # Number of parallel CPU workers
        self.prefetch_factor = 2  # Number of batches to prefetch
        
        # Initialize models with mixed precision
        self.vae = VAEWrapper(self.device, config).half()
        self.clip = CLIPTextEncoder(self.device, config).half()
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device).half().eval()
        
        # Create feature directories aggressively
        self.feature_dir = Path(config.feature_cache_path)
        self._force_create_dirs()

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

    def process_batch(self, batch_paths):
        """Process a batch of images in parallel"""
        batch_data = []
        valid_paths = []
        
        # Parallel loading and validation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._load_and_validate, p): p for p in batch_paths}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_paths.append(result[0])
                    batch_data.append(result[1])
        
        if not valid_paths:
            return 0

        # Batch processing
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Process VAE in batch
            img_tensors = torch.cat([d['img'] for d in batch_data])
            latents = self.vae.encode(img_tensors).cpu()
            
            # Process CLIP in batch
            texts = [d['text'] for d in batch_data]
            clip_embeds = self.clip.encode(texts).cpu()
            
            # Process DINO in batch
            dino_imgs = torch.cat([d['dino_img'] for d in batch_data])
            dino_features = self.dino(dino_imgs).cpu()
        
        # Parallel saving
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for i, path in enumerate(valid_paths):
                features = {
                    'latent': latents[i],
                    'clip': clip_embeds[i],
                    'dino': dino_features[i],
                    'dims': batch_data[i]['dims'],
                    'bucket': batch_data[i]['bucket']
                }
                executor.submit(self._save_features, path, features)
        
        return len(valid_paths)

    def _load_and_validate(self, img_path):
        """Parallel loading and preprocessing"""
        try:
            caption_path = Path(img_path).with_suffix('.txt')
            if not caption_path.exists():
                return None

            # Load and preprocess image
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                orig_w, orig_h = img.size
                
                # VAE preprocessing
                vae_img = transforms.ToTensor()(img.resize((512, 512))).unsqueeze(0)
                
                # DINO preprocessing
                dino_img = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(img).unsqueeze(0)

            return (
                img_path,
                {
                    'img': vae_img,
                    'text': caption_path.read_text(),
                    'dino_img': dino_img,
                    'dims': torch.tensor([orig_w, orig_h], dtype=torch.int16),
                    'bucket': self._get_bucket_index(orig_w, orig_h)
                }
            )
        except Exception as e:
            return None

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
    # Initialize distributed processing with explicit NCCL settings
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    
    # Configure NCCL environment variables
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '600000'  # 10 minute timeout
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:54321',  # Explicit TCP init
        world_size=int(os.environ['WORLD_SIZE']),
        rank=rank,
        timeout=timedelta(minutes=90)  # Increased timeout
    )
    
    config = get_config()
    processor = FeatureGenerator(config)
    
    # Get all images without validation
    all_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Distribute images evenly with proper device awareness
    chunk = len(all_images) // dist.get_world_size()
    local_images = all_images[rank*chunk : (rank+1)*chunk]
    
    # Create optimized data loader
    dataset = ImageDataset(local_images)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=processor.batch_size,
        sampler=sampler,
        num_workers=processor.num_workers,
        prefetch_factor=processor.prefetch_factor,
        pin_memory=True
    )
    
    # Process with batched pipeline
    with tqdm(total=len(dataset), desc=f"GPU {rank}", position=rank) as pbar:
        for batch in loader:
            processed = processor.process_batch(batch)
            pbar.update(processed)
    
    # Synchronize with device specification
    if dist.get_world_size() > 1:
        dist.barrier(device_ids=[rank], async_op=False)  # Explicit device sync
    
    # Cluster on rank 0 with error handling
    if rank == 0:
        try:
            processor.run_clustering()
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            dist.destroy_process_group()
            raise
    
    # Graceful shutdown
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 