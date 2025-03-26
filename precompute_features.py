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

    def process_image(self, img_path):
        """Batch image processing"""
        self.image_buffer.append(img_path)
        if len(self.image_buffer) >= self.batch_size:
            self._process_batch()
            
    def _process_batch(self):
        """Process a batch of images in parallel"""
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._single_process, path) 
                      for path in self.image_buffer]
            for future in as_completed(futures):
                future.result()  # Handle exceptions here
        self.image_buffer.clear()

    def _single_process(self, img_path):
        """Process a single image using filenames for latent storage"""
        try:
            # Check for caption file first
            img_path_obj = Path(img_path)
            caption_path = img_path_obj.with_suffix('.txt')
            if not caption_path.exists():
                print(f"Rank {self.rank}: No caption found for {img_path}")
                return False
            
            # Use the filename as the base for latent storage
            base_name = img_path_obj.stem
            
            # Read caption text
            caption_text = caption_path.read_text()
            
            # Process the image
            with Image.open(img_path) as img:
                # Handle corrupt images and various color modes
                img = img.convert('RGB')  # Force RGB conversion for all images
                orig_w, orig_h = img.size
                
                # Calculate nearest bucket
                bucket_idx = self._get_bucket_index(orig_w, orig_h)
                features = {
                    'latent': self._extract_vae_latent(img) if 'vae' in self.enabled_features else None,
                    'clip': self._extract_clip_embedding(caption_text) if 'clip' in self.enabled_features else None,
                    't5': self._extract_t5_embedding(caption_text) if 't5' in self.enabled_features else None,
                    'dino': self._extract_dino_features(img) if 'dino' in self.enabled_features and self.dino is not None else None,
                    'dims': torch.tensor(img.size, dtype=torch.int16) if 'dims' in self.enabled_features else None,
                    'bucket': torch.tensor(bucket_idx, dtype=torch.int16) if 'buckets' in self.enabled_features else None
                }
            
            # Save with rank-specific naming, only save enabled features
            features = {k: v for k, v in features.items() if v is not None}
            self._save_features(base_name, features)
            
            # Log successful processing
            if self.rank == 0 and random.random() < 0.01:  # Log ~1% of successful files on rank 0
                print(f"Rank {self.rank}: Successfully processed {img_path}, saved {len(features)} features")
            
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

    def _save_features(self, base_name, features):
        """Force-save features with rank ID"""
        saved_files = []
        
        # Create a reverse mapping from feature keys to feature types
        key_to_type = {v['feature_key']: k for k, v in self.FEATURE_PROCESSORS.items()}
        
        for feat_key, data in features.items():
            if feat_key in key_to_type and key_to_type[feat_key] in self.enabled_features:
                feat_type = key_to_type[feat_key]
                save_dir = self.feature_dir/self.FEATURE_PROCESSORS[feat_type]['save_prefix']
                save_path = save_dir/f"{base_name}_rank{self.rank}.pt"
                
                try:
                    # Ensure directory exists
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save data
                    torch.save(data, save_path)
                    saved_files.append(str(save_path))
                except Exception as e:
                    print(f"Rank {self.rank} failed to save {save_path}: {str(e)}")
        
        # Occasional logging of saved files
        if self.rank == 0 and random.random() < 0.005:  # Log ~0.5% of files on rank 0
            print(f"Rank {self.rank}: Saved files: {saved_files}")
        
        return len(saved_files) > 0

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
            'handler': '_extract_vae_latent',
            'save_prefix': 'latents',
            'feature_key': 'latent'  # The key used in the features dictionary
        },
        'clip': {
            'handler': '_extract_clip_embedding',
            'save_prefix': 'clip',
            'feature_key': 'clip'
        },
        't5': {
            'handler': '_extract_t5_embedding',
            'save_prefix': 't5',
            'feature_key': 't5'
        },
        'dino': {
            'handler': '_extract_dino_features', 
            'save_prefix': 'dino_features',
            'feature_key': 'dino'
        },
        'buckets': {
            'handler': '_extract_bucket_index',
            'save_prefix': 'buckets',
            'feature_key': 'bucket'  # This is the critical mapping!
        },
        'dims': {
            'handler': '_extract_dims',
            'save_prefix': 'dims',
            'feature_key': 'dims'
        }
    }

    # Add a new wrapper function to extract bucket index from an image path
    def _extract_bucket_index(self, img_path):
        """Extract bucket index from image dimensions"""
        with Image.open(img_path) as img:
            width, height = img.size
            return torch.tensor(self._get_bucket_index(width, height), dtype=torch.int16)

def main():
    # Parse command line arguments
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
        feature_map = {
            'vae-latents': 'vae',
            'clip-latents': 'clip', 
            't5-latents': 't5',
            'dino-features': 'dino',
            'buckets': 'buckets',  # This maps directly from arg name to feature type
            'dims': 'dims',
            'clustering': 'clustering'
        }
        enabled_features = [feature_map[f] for f in vars(args) if vars(args)[f] and f in feature_map]

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
    
    # Get all images
    if rank == 0:
        print("Scanning dataset directory...")
    all_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Distribute images with remainder handling
    world_size = dist.get_world_size()
    chunk_size = len(all_images) // world_size
    remainder = len(all_images) % world_size
    
    start = rank * chunk_size + min(rank, remainder)
    end = (rank + 1) * chunk_size + min(rank + 1, remainder)
    local_images = all_images[start:end]
    
    print(f"Rank {rank} received {len(local_images)} images to process")
    
    # Define fixed processing order
    processing_order = ['vae', 'clip', 't5', 'dino', 'buckets', 'dims']

    # Only process features that are enabled
    features_to_process = [f for f in processing_order if f in enabled_features]
    
    # Process one feature type at a time for all images
    for feature_type in features_to_process:
        if rank == 0:
            print(f"All ranks processing feature type: {feature_type}")
        
        # Initialize feature generator with just this feature type
        processor = FeatureGenerator(config, [feature_type])
        
        # This is the key fix - make sure we use the correct save directory
        save_prefix = processor.FEATURE_PROCESSORS[feature_type]['save_prefix']
        save_dir = processor.feature_dir/save_prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch size depends on feature type - use larger batches for smaller features
        if feature_type in ['buckets', 'dims']:
            batch_size = 64
        else:
            batch_size = 16  # For larger models like VAE, DINO
            
        try:
            # Main processing loop with batching
            with tqdm(total=len(local_images), desc=f"GPU {rank} - {feature_type}", position=rank) as pbar:
                for batch_start in range(0, len(local_images), batch_size):
                    batch_end = min(batch_start + batch_size, len(local_images))
                    batch_paths = local_images[batch_start:batch_end]
                    
                    # Track successfully processed items for pbar update
                    processed_count = 0
                    
                    # Check which files already exist
                    batch_to_process = []
                    for img_path in batch_paths:
                        img_path_obj = Path(img_path)
                        base_name = img_path_obj.stem
                        save_path = save_dir/f"{base_name}_rank{rank}.pt"
                        
                        # Skip if already processed
                        if save_path.exists():
                            processed_count += 1
                            continue
                            
                        # For text features, check if caption exists
                        if feature_type in ['clip', 't5']:
                            caption_path = img_path_obj.with_suffix('.txt')
                            if not caption_path.exists():
                                processed_count += 1
                                continue
                                
                        batch_to_process.append(img_path)
                    
                    # Skip if all files already processed
                    if not batch_to_process:
                        pbar.update(processed_count)
                        continue
                        
                    # Process remaining batch
                    for img_path in batch_to_process:
                        try:
                            img_path_obj = Path(img_path)
                            base_name = img_path_obj.stem
                            
                            # Process based on feature type
                            try:
                                if feature_type in ['clip', 't5']:
                                    # Text feature processing
                                    caption_path = img_path_obj.with_suffix('.txt')
                                    caption_text = caption_path.read_text()
                                    
                                    if feature_type == 'clip':
                                        feature_data = processor._extract_clip_embedding(caption_text)
                                    else:  # t5
                                        feature_data = processor._extract_t5_embedding(caption_text)
                                else:
                                    # Image feature processing
                                    with Image.open(img_path) as img:
                                        img = img.convert('RGB')  # Ensure RGB format
                                        
                                        if feature_type == 'vae':
                                            feature_data = processor._extract_vae_latent(img)
                                        elif feature_type == 'dino':
                                            feature_data = processor._extract_dino_features(img)
                                        elif feature_type == 'buckets':
                                            width, height = img.size
                                            feature_data = torch.tensor(processor._get_bucket_index(width, height), dtype=torch.int16)
                                        elif feature_type == 'dims':
                                            feature_data = processor._extract_dims(img)
                                
                                # Save the feature
                                save_path = save_dir/f"{base_name}_rank{rank}.pt"
                                torch.save(feature_data, save_path)
                                
                                # Occasional logging
                                if rank == 0 and random.random() < 0.001:
                                    print(f"Rank {rank}: Saved {feature_type} for {base_name}")
                                    
                                processed_count += 1
                                
                            except Exception as e:
                                print(f"Rank {rank} failed processing {feature_type} for {img_path}: {str(e)}")
                                processed_count += 1  # Still count as processed for progress
                                continue
                                
                        except Exception as e:
                            print(f"Rank {rank} failed completely for {img_path}: {str(e)}")
                            processed_count += 1
                            continue
                            
                    # Update progress bar with batch results
                    pbar.update(processed_count)
                        
            # Synchronize after each feature type
            dist.barrier()
            
            # Free memory
            del processor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Rank {rank} failed during {feature_type} processing: {str(e)}")
            dist.barrier(timeout=60)
            continue

    # Clustering code with proper status updates
    if 'clustering' in enabled_features:
        cluster_start_time = time.time()
        cluster_error = torch.tensor([0], device=device)
        clustering_status = torch.tensor([0], device=device)
        
        # Initialize processor for clustering
        cluster_processor = FeatureGenerator(config, ['clustering'])
        feature_dir = cluster_processor.feature_dir  # Define feature_dir properly
        
        if rank == 0:
            try:
                # Clear previous markers
                (feature_dir/"clusters/started.pt").unlink(missing_ok=True)
                (feature_dir/"clusters/error.pt").unlink(missing_ok=True)
                
                # Signal start
                (feature_dir/"clusters/started.pt").touch()
                
                # Run clustering
                cluster_processor.run_clustering()
                
                # Signal completion on success
                clustering_status.fill_(2)
                print("Clustering completed successfully")
                
            except Exception as e:
                print(f"Clustering failed: {str(e)}")
                cluster_error.fill_(1)
                clustering_status.fill_(1)
                (feature_dir/"clusters/error.pt").touch()
        
        # Broadcast status to all ranks
        dist.broadcast(cluster_error, src=0)
        dist.broadcast(clustering_status, src=0)
        
        # Check for immediate error
        if cluster_error.item() == 1:
            raise RuntimeError("Clustering failed on rank 0")
        
        # Wait for completion or timeout if not already done
        if clustering_status.item() != 2:
            while True:
                # Poll status
                dist.all_reduce(clustering_status, op=dist.ReduceOp.MAX)
                
                if clustering_status.item() == 2:
                    break  # Completed
                elif clustering_status.item() == 1:
                    raise RuntimeError("Clustering failed after initial check")
                
                if time.time() - cluster_start_time > 7200:  # 2 hour timeout
                    raise RuntimeError("Clustering timed out")
                
                # Status reporting
                if rank == 0:
                    print(f"Clustering progress: {time.time() - cluster_start_time:.1f}s elapsed")
                
                time.sleep(5)

    # Add GPU health check
    test_tensor = torch.randn(1024, device=device)
    torch.cuda.synchronize()
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

    # Final cleanup
    feature_dir = Path(config.feature_cache_path)  # Define feature_dir for cleanup
    if rank == 0:
        for r in range(dist.get_world_size()):
            (feature_dir/f"status_rank{r}.pt").unlink(missing_ok=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        # Force exit with error code
        import sys
        sys.exit(1) 