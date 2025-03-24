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

    def _force_create_dirs(self):
        """Overwrite any existing feature directories"""
        dirs = ['latents', 'clip', 'clusters', 'dims', 'dino_features', 'buckets']
        for d in dirs:
            dir_path = self.feature_dir/d
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)

    def process_image(self, img_path):
        """Handle mixed precision and missing captions"""
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
        img_tensor = transforms.ToTensor()(img.resize((512, 512))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vae.encode(img_tensor).cpu()

    def _extract_clip_embedding(self, caption):
        """Raw text embedding without validation"""
        with torch.no_grad():
            return self.clip.encode([caption]).cpu()

    def _extract_dino_features(self, img):
        """DINO feature extraction"""
        prep_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.dino(prep_img).cpu()

    def _save_features(self, base_name, features):
        """Force-save features with rank ID"""
        torch.save(features['latent'], self.feature_dir/f"latents/{base_name}_rank{self.rank}.pt")
        torch.save(features['clip'], self.feature_dir/f"clip/{base_name}_rank{self.rank}.pt")
        torch.save(features['dino'], self.feature_dir/f"dino_features/{base_name}_rank{self.rank}.pt")
        torch.save(features['dims'], self.feature_dir/f"dims/{base_name}_rank{self.rank}.pt")
        torch.save(features['bucket'], self.feature_dir/f"buckets/{base_name}_rank{self.rank}.pt")

    def run_clustering(self):
        """Global clustering after feature collection"""
        # Load all features across ranks
        features = []
        for f in (self.feature_dir/"dino_features").glob("*.pt"):
            features.append(torch.load(f))
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
    # Initialize distributed processing
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = get_config()
    processor = FeatureGenerator(config)
    
    # Get all images without validation
    all_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Distribute images evenly
    chunk = len(all_images) // world_size
    local_images = all_images[rank*chunk : (rank+1)*chunk]
    
    # Process with progress
    with tqdm(total=len(local_images), desc=f"Rank {rank}", position=rank) as pbar:
        for img_path in local_images:
            if processor.process_image(img_path):
                pbar.update(1)
    
    # Cluster on rank 0
    dist.barrier()
    if rank == 0:
        processor.run_clustering()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 