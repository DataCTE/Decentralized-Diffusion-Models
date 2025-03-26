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
from queue import Queue
from threading import Lock

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

    def _process_image_caption_pair_with_uuid(self, img_path, caption_path, base_name):
        """Process a single image-caption pair with pre-determined UUID"""
        try:
            # First check if this feature already exists for the enabled features
            feature_dir = self.feature_dir
            missing_features = []
            for feat in self.enabled_features:
                if feat in self.FEATURE_PROCESSORS:
                    save_prefix = self.FEATURE_PROCESSORS[feat]['save_prefix']
                    filename = f"{base_name}_rank{self.rank}.pt"
                    if not (feature_dir/save_prefix/filename).exists():
                        missing_features.append(feat)
            
            # If all features exist, skip processing
            if not missing_features:
                return True
            
            # Read caption text only once if needed
            caption_text = None
            if 'clip' in missing_features or 't5' in missing_features:
                caption_text = Path(caption_path).read_text()
            
            # Load image only if needed
            img = None
            if any(f in missing_features for f in ['vae', 'dino', 'dims', 'buckets']):
                img = Image.open(img_path).convert('RGB')
            
            # Extract only the missing features
            features = {}
            
            if 'vae' in missing_features:
                features['vae'] = self._extract_vae_latent(img)
            
            if 'clip' in missing_features:
                features['clip'] = self._extract_clip_embedding(caption_text)
            
            if 't5' in missing_features:
                features['t5'] = self._extract_t5_embedding(caption_text)
            
            if 'dino' in missing_features:
                features['dino'] = self._extract_dino_features(img)
            
            if 'dims' in missing_features:
                features['dims'] = torch.tensor(img.size, dtype=torch.int16)
            
            if 'buckets' in missing_features:
                features['bucket'] = torch.tensor(self._get_bucket_index(img.width, img.height), dtype=torch.int16)
            
            # Save only the missing features
            for feat, data in features.items():
                if feat in self.enabled_features:
                    save_prefix = self.FEATURE_PROCESSORS[feat]['save_prefix']
                    save_path = feature_dir/save_prefix/f"{base_name}_rank{self.rank}.pt"
                    torch.save(data, save_path)
            
            # Close image if opened
            if img is not None:
                img.close()
            
            return True
        
        except Exception as e:
            logger.error(f"Rank {self.rank} failed processing {img_path}: {str(e)}")
            return False

    def _check_existing_features(self, feature_type):
        """List all existing features of a certain type"""
        dir_path = self.feature_dir / self.FEATURE_PROCESSORS[feature_type]['save_prefix']
        if not dir_path.exists():
            return set()
        
        # Get all UUIDs from files (removing _rank suffix)
        pattern = f"*_rank{self.rank}.pt"
        return {p.stem.split('_rank')[0] for p in dir_path.glob(pattern)}

    def _extract_t5_embeddings_batch(self, captions):
        """Extract T5 embeddings for a batch of captions with enhanced performance"""
        # Use mixed precision for faster processing
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Add timing benchmarks to detect bottlenecks
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                # Process in sub-batches if needed for very large batches
                if len(captions) > 256:
                    all_embeddings = []
                    for i in range(0, len(captions), 256):
                        sub_captions = captions[i:i+256]
                        sub_embeddings = self.t5.encode(sub_captions)
                        all_embeddings.append(sub_embeddings.cpu())
                    embeddings = torch.cat(all_embeddings, dim=0)
                else:
                    embeddings = self.t5.encode(captions).cpu()
                
                end.record()
                torch.cuda.synchronize()
                
                # Measure and log inference time for performance tuning
                if len(captions) > 32:  # Only log for larger batches
                    inference_ms = start.elapsed_time(end)
                    logger.debug(f"T5 embedding time for {len(captions)} captions: {inference_ms:.2f}ms " 
                               f"({inference_ms/len(captions):.2f}ms per caption)")
                
                return embeddings

    def _save_t5_embedding(self, uuid_str, embedding):
        """Save a T5 embedding with the given UUID"""
        save_path = self.feature_dir / "t5" / f"{uuid_str}_rank{self.rank}.pt"
        torch.save(embedding, save_path)
        return True

def _process_directory(config, rank=0, world_size=1, only_missing=True):
    """Efficiently scan and divide dataset for distributed processing with special handling for T5 embeddings"""
    dataset_path = Path(config.dataset_path)
    feature_path = Path(config.feature_cache_path)
    
    # For very large datasets, we need to avoid loading entire file listings into memory
    # Only have rank 0 scan the dataset once
    if rank == 0:
        logger.info(f"Scanning dataset directory: {dataset_path}")
        
        # When only T5 embeddings are enabled, use a more targeted approach
        if config.enabled_features == ['t5']:
            logger.info("Targeting T5 embeddings specifically...")
            
            # Look for CLIP embeddings to derive T5 requirement from
            clip_dir = feature_path / "clip"
            t5_dir = feature_path / "t5"
            
            if not clip_dir.exists():
                logger.error("CLIP embeddings directory not found - needed as reference for T5 embeddings")
                raise FileNotFoundError(f"CLIP directory {clip_dir} not found")
                
            # Make sure T5 directory exists
            t5_dir.mkdir(exist_ok=True)
            
            # Get existing T5 UUIDs as a set for fast membership testing
            t5_pattern = "*_rank*.pt" 
            t5_uuids = set()
            
            # Process in chunks to avoid memory explosion with millions of files
            if t5_dir.exists():
                logger.info("Scanning existing T5 embeddings...")
                # Count files efficiently without loading all into memory
                t5_count = 0
                for batch in _chunked_glob(t5_dir, t5_pattern, 10000):
                    # Extract UUIDs without rank suffix
                    t5_uuids.update(p.stem.split('_rank')[0] for p in batch)
                    t5_count += len(batch)
                
                logger.info(f"Found {len(t5_uuids)} existing T5 UUIDs across {t5_count} files")
            
            # Scan CLIP embeddings - don't try to load all files at once
            logger.info("Scanning CLIP embeddings to find needed T5 processing...")
            clip_uuids = set()
            clip_total = 0
            
            # Create caption buffers
            captions_to_process = []
            uuids_to_process = []
            
            # Go through clip files in batches
            with tqdm(desc="Scanning CLIP files") as pbar:
                for batch in _chunked_glob(clip_dir, "*_rank*.pt", 10000):
                    # Update progress for all files scanned
                    pbar.update(len(batch))
                    clip_total += len(batch)
                    
                    # Get UUIDs from this batch
                    batch_uuids = [p.stem.split('_rank')[0] for p in batch]
                    
                    # Check which ones need T5 embeddings
                    for uuid_str in batch_uuids:
                        # Only consider UUIDs we haven't seen before
                        if uuid_str not in clip_uuids:
                            clip_uuids.add(uuid_str)
                            
                            # Check if T5 embedding already exists
                            if uuid_str not in t5_uuids or not args.force_recompute:
                                caption = _get_caption_from_mapping(config, uuid_str)
                                
                                if caption:
                                    uuids_to_process.append(uuid_str)
                                    captions_to_process.append(caption)
                                else:
                                    # Try to find the caption file directly if no mapping
                                    img_path, caption_path = None, None
                                    
                                    # Check if we have a metadata directory with image paths
                                    meta_dir = feature_path / "metadata"
                                    if meta_dir.exists():
                                        meta_file = meta_dir / f"{uuid_str}.json"
                                        if meta_file.exists():
                                            import json
                                            with open(meta_file) as f:
                                                metadata = json.load(f)
                                                if "caption_path" in metadata:
                                                    caption_path = metadata["caption_path"]
                                    
                                    # If no path found but we have the dataset, look for the image
                                    if caption_path is None:
                                        # This is a fallback and slower - prints a warning
                                        logger.warning(f"No caption mapping found for {uuid_str}, "
                                                     f"looking in dataset directly (slow)")
                                        
                                        # Try to get the caption from the dataset directly
                                        for img_ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                            # Look in first level directories of dataset
                                            for img_dir in dataset_path.iterdir():
                                                if img_dir.is_dir():
                                                    # Try some common patterns
                                                    for pattern in [f"*{uuid_str}*{img_ext}", f"*{img_ext}"]:
                                                        matches = list(img_dir.glob(pattern))
                                                        if matches:
                                                            for img_file in matches:
                                                                caption_file = img_file.with_suffix('.txt')
                                                                if caption_file.exists():
                                                                    caption_path = caption_file
                                                                    break
                                                            if caption_path:
                                                                break
                                                if caption_path:
                                                    break
                                            if caption_path:
                                                break
                                    
                                    # If we found a caption file, read it
                                    if caption_path and Path(caption_path).exists():
                                        try:
                                            caption = Path(caption_path).read_text()
                                            uuids_to_process.append(uuid_str)
                                            captions_to_process.append(caption)
                                        except Exception as e:
                                            logger.error(f"Error reading caption file {caption_path}: {e}")
            
            logger.info(f"Scanned {clip_total} CLIP files with {len(clip_uuids)} unique UUIDs")
            logger.info(f"Found {len(uuids_to_process)} items needing T5 embeddings")
            
            # Save the captions and uuids for distributed processing
            if not uuids_to_process:
                logger.warning("No files found needing T5 embeddings! Check if:")
                logger.warning("  1. All T5 embeddings already exist (use --force-recompute to regenerate)")
                logger.warning("  2. CLIP embeddings exist in the expected location")
                logger.warning("  3. Caption mapping or dataset structure is correctly configured")
            
            # Save for distributed processing
            with open(f"{config.feature_cache_path}/t5_task.pkl", 'wb') as f:
                pickle.dump((uuids_to_process, captions_to_process), f)
                
            # Flag that we're in captions-only mode
            with open(f"{config.feature_cache_path}/captions_only_mode", 'w') as f:
                f.write("1")
        else:
            # Fallback to standard directory scan
            # ... [existing directory scanning code] ...
            logger.info("Performing full dataset scan...")
            
            # If only processing missing files, gather existing UUIDs efficiently
            existing_uuids = {}
            if only_missing:
                for feat_type in config.enabled_features:
                    if feat_type in ['vae', 'clip', 't5', 'dino']:
                        feat_dir = feature_path / FeatureGenerator.FEATURE_PROCESSORS[feat_type]['save_prefix']
                        if feat_dir.exists():
                            # Count files instead of loading full list
                            count = sum(1 for _ in feat_dir.glob('*_rank*.pt'))
                            logger.info(f"Found {count} existing {feat_type} files")
                            
                            # Only load UUIDs for feature types we need to check against
                            if feat_type in config.enabled_features:
                                # Process in chunks to avoid memory issues
                                existing_uuids[feat_type] = set()
                                for files in _chunked_glob(feat_dir, "*_rank*.pt", 10000):
                                    existing_uuids[feat_type].update(p.stem.split('_rank')[0] for p in files)
            
            # Process dataset incrementally in chunks
            image_paths = []
            caption_paths = []
            uuids = []
            
            # For very large datasets, avoid loading all paths into memory at once
            # Instead, scan directories in chunks and process incrementally
            valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
            
            # Use more efficient file discovery
            image_files = _find_image_files_with_captions(dataset_path, valid_exts)
            total_images = len(image_files)
            logger.info(f"Found {total_images} total image files with captions")
            
            # Process in chunks to avoid memory issues
            with tqdm(total=total_images, desc="Checking files") as pbar:
                for chunk_start in range(0, total_images, 10000):
                    chunk_end = min(chunk_start + 10000, total_images)
                    chunk = image_files[chunk_start:chunk_end]
                    
                    # Process this chunk
                    for img_path in chunk:
                        caption_path = Path(img_path).with_suffix('.txt')
                        
                        # Generate UUID from file content hashes - but do it efficiently
                        # For large datasets, computing MD5 of every file is expensive
                        # Use a combination of path and mtime for faster processing
                        img_stat = os.stat(img_path)
                        cap_stat = os.stat(caption_path)
                        hash_input = f"{img_path}:{img_stat.st_size}:{img_stat.st_mtime}:{caption_path}:{cap_stat.st_size}:{cap_stat.st_mtime}"
                        file_uuid = hashlib.md5(hash_input.encode()).hexdigest()
                        
                        # Check if we need to process this file
                        needs_processing = False
                        if not only_missing:
                            needs_processing = True
                        else:
                            # Check if any required feature is missing
                            for feat_type in config.enabled_features:
                                if feat_type in ['vae', 'clip', 't5', 'dino']:
                                    if (feat_type not in existing_uuids or 
                                        file_uuid not in existing_uuids[feat_type]):
                                        needs_processing = True
                                        break
                        
                        if needs_processing:
                            image_paths.append(img_path)
                            caption_paths.append(str(caption_path))
                            uuids.append(file_uuid)
                    
                    pbar.update(len(chunk))
            
            logger.info(f"Found {len(image_paths)} image-caption pairs requiring processing")
            
            # Save the paths to a temporary file for other ranks to read
            with open(f"{config.feature_cache_path}/image_paths.pkl", 'wb') as f:
                pickle.dump((image_paths, caption_paths, uuids), f)
    
    # Synchronize to ensure rank 0 has finished writing the file
    if world_size > 1:
        dist.barrier()
    
    # Check if we're in captions-only mode for T5
    captions_only_mode = os.path.exists(f"{config.feature_cache_path}/captions_only_mode")
    
    if captions_only_mode:
        # Load the T5 task file
        with open(f"{config.feature_cache_path}/t5_task.pkl", 'rb') as f:
            uuids, caption_texts = pickle.load(f)
        
        # Distribute task among ranks
        total_items = len(uuids)
        items_per_rank = total_items // world_size
        remainder = total_items % world_size
        
        # Calculate this rank's start and end indices with balanced distribution
        start_idx = rank * items_per_rank + min(rank, remainder)
        end_idx = start_idx + items_per_rank + (1 if rank < remainder else 0)
        
        # Get this rank's subset of captions
        rank_uuids = uuids[start_idx:end_idx]
        rank_captions = caption_texts[start_idx:end_idx]
        
        logger.info(f"Rank {rank} processing {len(rank_uuids)} captions for T5 embedding in direct mode")
        
        # Clean up if last rank
        if rank == world_size - 1:
            if os.path.exists(f"{config.feature_cache_path}/t5_task.pkl"):
                os.remove(f"{config.feature_cache_path}/t5_task.pkl")
            if os.path.exists(f"{config.feature_cache_path}/captions_only_mode"):
                os.remove(f"{config.feature_cache_path}/captions_only_mode")
        
        return None, None, rank_uuids, rank_captions
    else:
        # All ranks read the paths file - standard mode
        with open(f"{config.feature_cache_path}/image_paths.pkl", 'rb') as f:
            image_paths, caption_paths, uuids = pickle.load(f)
        
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
        rank_uuids = uuids[start_idx:end_idx]
        
        logger.info(f"Rank {rank} processing {len(rank_image_paths)} image-caption pairs")
        
        # Clean up the paths file if we're the last rank
        if rank == world_size - 1 and os.path.exists(f"{config.feature_cache_path}/image_paths.pkl"):
            os.remove(f"{config.feature_cache_path}/image_paths.pkl")
        
        return rank_image_paths, rank_caption_paths, rank_uuids, None

# Add these helper functions for efficient file handling

def _chunked_glob(path, pattern, chunk_size=10000):
    """Yield chunks of glob results to avoid memory issues"""
    all_files = list(path.glob(pattern))
    for i in range(0, len(all_files), chunk_size):
        yield all_files[i:i + chunk_size]

def _find_image_files_with_captions(base_path, valid_exts):
    """Find all image files that have accompanying caption files"""
    result = []
    
    # Use os.walk for better performance
    for root, _, files in os.walk(base_path):
        root_path = Path(root)
        
        # First collect all image files
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in valid_exts)]
        
        # Then filter to those with captions
        for img_file in image_files:
            img_path = root_path / img_file
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                result.append(str(img_path))
    
    return result

def _find_source_files(config, uuids):
    """Find source image and caption files from UUIDs using a mapping file if available"""
    # If we have a mapping file, use it
    if hasattr(config, "uuid_mapping_file") and os.path.exists(config.uuid_mapping_file):
        return _lookup_files_from_mapping(config, uuids)
    
    # Otherwise fall back to scanning the dataset
    image_paths = []
    caption_paths = []
    
    for uuid_str in uuids:
        # Find files by UUID
        for feat_dir in ["latents", "clip", "dims"]:
            dir_path = Path(config.feature_cache_path) / feat_dir
            if not dir_path.exists():
                continue
                
            # Look for any file matching this UUID
            match_files = list(dir_path.glob(f"{uuid_str}_rank*.pt"))
            if match_files:
                # Try to use the metadata file if it exists
                meta_file = Path(config.feature_cache_path) / "metadata" / f"{uuid_str}.json"
                if meta_file.exists():
                    import json
                    with open(meta_file) as f:
                        metadata = json.load(f)
                        if "image_path" in metadata and "caption_path" in metadata:
                            image_paths.append(metadata["image_path"])
                            caption_paths.append(metadata["caption_path"])
                            break
    
    # If we couldn't find them, leave empty
    missing = len(uuids) - len(image_paths)
    if missing > 0:
        logger.warning(f"Could not find source files for {missing} UUIDs")
    
    return image_paths, caption_paths

def _get_caption_from_mapping(config, uuid_str):
    """Get caption text directly from mapping file"""
    if hasattr(config, "caption_mapping_file") and os.path.exists(config.caption_mapping_file):
        import json
        with open(config.caption_mapping_file) as f:
            mapping = json.load(f)
            return mapping.get(uuid_str, None)
    return None

def _lookup_files_from_mapping(config, uuids):
    """Look up image and caption paths from mapping file"""
    import json
    with open(config.uuid_mapping_file) as f:
        mapping = json.load(f)
    
    image_paths = []
    caption_paths = []
    
    for uuid_str in uuids:
        if uuid_str in mapping:
            entry = mapping[uuid_str]
            image_paths.append(entry["image_path"])
            caption_paths.append(entry["caption_path"])
    
    return image_paths, caption_paths

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
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation of all features')
    parser.add_argument('--use-existing-dino', action='store_true',
                        help='Use existing DINO features from disk')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for I/O operations')
    parser.add_argument('--direct-captions', action='store_true',
                        help='Use direct caption processing for T5 (faster)')
    parser.add_argument('--t5-threads', type=int, default=3,
                        help='Number of threads for T5 processing pipeline')
    parser.add_argument('--t5-batch', type=int, default=256,
                        help='Manual override for T5 batch size')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for troubleshooting')
    parser.add_argument('--regen-t5', action='store_true',
                        help='Regenerate T5 embeddings from existing CLIP files')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose output including processed files')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, 
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
    config.enabled_features = enabled_features  # Store for later use

    # Update batch size from args
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    # Verify dataset path exists
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path {config.dataset_path} not found")
    
    # Create feature cache directory if it doesn't exist
    os.makedirs(config.feature_cache_path, exist_ok=True)
    
    # Special handling for T5 regeneration from CLIP
    if args.regen_t5 and 't5' in enabled_features:
        if rank == 0:
            logger.info("Setting up for T5 regeneration from CLIP embeddings")
            clip_dir = Path(config.feature_cache_path) / "clip"
            t5_dir = Path(config.feature_cache_path) / "t5"
            
            # Create T5 directory if it doesn't exist
            t5_dir.mkdir(exist_ok=True)
            
            if not clip_dir.exists():
                logger.error("CLIP directory not found. Cannot generate T5 embeddings.")
                raise FileNotFoundError(f"CLIP directory {clip_dir} not found")
            
            # Get all CLIP UUIDs
            clip_uuids = []
            clip_files = list(clip_dir.glob("*.pt"))
            logger.info(f"Found {len(clip_files)} CLIP files")
            
            # Extract UUIDs from filenames
            for clip_file in clip_files:
                uuid_str = clip_file.stem.split('_rank')[0]
                clip_uuids.append(uuid_str)
                
            # Check if T5 files already exist
            existing_t5_uuids = set()
            if not args.force_recompute and t5_dir.exists():
                for t5_file in t5_dir.glob("*.pt"):
                    existing_t5_uuids.add(t5_file.stem.split('_rank')[0])
                    
            # Filter out UUIDs that already have T5 embeddings
            if not args.force_recompute:
                clip_uuids = [uuid for uuid in clip_uuids if uuid not in existing_t5_uuids]
                
            logger.info(f"Found {len(clip_uuids)} CLIP files needing T5 embeddings")
            
            # Load captions for these UUIDs
            captions = []
            for uuid_str in clip_uuids:
                # Try to get caption from mapping
                caption = _get_caption_from_mapping(config, uuid_str)
                if caption:
                    captions.append(caption)
                else:
                    logger.warning(f"No caption found for {uuid_str}, using empty placeholder")
                    captions.append("")  # Use empty caption as placeholder
            
            # Save for distributed processing
            with open(f"{config.feature_cache_path}/t5_task.pkl", 'wb') as f:
                pickle.dump((clip_uuids, captions), f)
                
            # Flag that we're in captions-only mode for T5
            with open(f"{config.feature_cache_path}/captions_only_mode", 'w') as f:
                f.write("1")
            
            logger.info(f"Prepared {len(clip_uuids)} captions for T5 embedding generation")
    
    # Get image-caption pairs efficiently, respecting force-recompute flag
    result = _process_directory(
        config, 
        rank, 
        world_size, 
        only_missing=not args.force_recompute
    )
    
    # Initialize feature generator
    processor = FeatureGenerator(config, enabled_features)
    processor.batch_size = batch_size  # Set batch size from command line
    
    # Check if we're in caption-only mode for T5
    captions_only_mode = len(result) == 4 and result[0] is None
    
    try:
        if captions_only_mode:
            # Process captions directly for T5 (much faster for large datasets)
            _, _, uuids, captions = result
            process_t5_embeddings_directly(processor, uuids, captions, batch_size, num_workers)
        else:
            # Standard processing mode
            image_paths, caption_paths, uuids, _ = result
            total_samples = len(image_paths)
            
            # Process image-text pairs in batches for better memory efficiency
            with tqdm(total=total_samples, desc=f"GPU {rank}", position=rank) as pbar:
                # Process in even larger batches for better GPU utilization
                for batch_idx in range(0, total_samples, batch_size * 4):
                    end_idx = min(batch_idx + batch_size * 4, total_samples)
                    
                    # Get a larger batch to keep the GPU busy
                    batch_image_paths = image_paths[batch_idx:end_idx]
                    batch_caption_paths = caption_paths[batch_idx:end_idx]
                    batch_uuids = uuids[batch_idx:end_idx]
                    
                    # Split into sub-batches for parallel processing
                    for sub_batch_idx in range(0, len(batch_image_paths), batch_size):
                        sub_end_idx = min(sub_batch_idx + batch_size, len(batch_image_paths))
                        
                        sub_img_paths = batch_image_paths[sub_batch_idx:sub_end_idx]
                        sub_cap_paths = batch_caption_paths[sub_batch_idx:sub_end_idx]
                        sub_uuids = batch_uuids[sub_batch_idx:sub_end_idx]
                        
                        # Process sub-batch in parallel
                        with ThreadPoolExecutor(max_workers=num_workers) as executor:
                            futures = [executor.submit(processor._process_image_caption_pair_with_uuid, 
                                                    img_path, cap_path, uuid_str) 
                                    for img_path, cap_path, uuid_str in zip(sub_img_paths, 
                                                                            sub_cap_paths, 
                                                                            sub_uuids)]
                            
                            # Process results as they complete
                            for future in as_completed(futures):
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(f"Error processing pair: {str(e)}")
                        
                        # Update progress
                        pbar.update(sub_end_idx - sub_batch_idx)
        
        # GPU health check
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

def process_t5_embeddings_directly(processor, uuids, captions, batch_size, num_workers):
    """Process T5 embeddings directly from caption texts with optimized multi-threading"""
    total_samples = len(uuids)
    logger.info(f"Processing {total_samples} T5 embeddings directly")
    
    # Determine optimal batch sizes based on GPU memory
    # T5 processing is very VRAM efficient so we can use larger batches
    gpu_mem = torch.cuda.get_device_properties(processor.device).total_memory / (1024**3)  # GB
    suggested_batch = min(512, int(gpu_mem * 48))  # Empirical formula for good T5 batch size
    t5_batch_size = suggested_batch - (suggested_batch % 64)  # Make divisible by 64
    
    logger.info(f"Using T5 batch size of {t5_batch_size} on GPU with {gpu_mem:.2f}GB memory")
    
    # Create a thread pool for I/O operations (saving)
    io_pool = ThreadPoolExecutor(max_workers=min(32, num_workers * 2))
    io_futures = []
    
    # Create a queue system for processing with multiple CPU threads
    caption_queue = Queue(maxsize=4)  # Limit queue size to prevent memory issues
    result_queue = Queue(maxsize=4)
    
    # Create a custom thread safe progress bar
    pbar_lock = Lock()
    pbar = tqdm(total=total_samples, desc=f"T5 GPU {processor.rank}", position=processor.rank)
    
    def enqueue_captions():
        """Thread that prepares batches and puts them in the queue"""
        for batch_idx in range(0, total_samples, t5_batch_size):
            end_idx = min(batch_idx + t5_batch_size, total_samples)
            
            batch_uuids = uuids[batch_idx:end_idx]
            batch_captions = captions[batch_idx:end_idx]
            
            # Split into smaller chunks for processing
            for i in range(0, len(batch_uuids), 128):
                chunk_end = min(i + 128, len(batch_uuids))
                
                chunk_uuids = batch_uuids[i:chunk_end]
                chunk_captions = batch_captions[i:chunk_end]
                
                caption_queue.put((chunk_uuids, chunk_captions))
        
        # Signal that we're done
        caption_queue.put(None)
    
    def process_captions():
        """Thread that processes captions through T5 model"""
        while True:
            item = caption_queue.get()
            if item is None:
                # Signal end of processing
                result_queue.put(None)
                break
                
            chunk_uuids, chunk_captions = item
            
            try:
                # Process on GPU
                with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision for speed
                    embeddings = processor._extract_t5_embeddings_batch(chunk_captions)
                
                # Push to result queue
                result_queue.put((chunk_uuids, embeddings))
            except Exception as e:
                logger.error(f"Error processing captions: {str(e)}")
                # Put a placeholder to keep counts aligned
                result_queue.put((chunk_uuids, None))
    
    def save_results():
        """Thread that saves results from the queue"""
        processed = 0
        while True:
            item = result_queue.get()
            if item is None:
                break
                
            chunk_uuids, embeddings = item
            if embeddings is None:
                with pbar_lock:
                    pbar.update(len(chunk_uuids))
                processed += len(chunk_uuids)
                continue
            
            # Save embeddings in parallel
            for j, uuid_str in enumerate(chunk_uuids):
                future = io_pool.submit(
                    processor._save_t5_embedding,
                    uuid_str,
                    embeddings[j]
                )
                io_futures.append(future)
            
            # Update progress
            with pbar_lock:
                pbar.update(len(chunk_uuids))
            processed += len(chunk_uuids)
    
    # Start the threads
    import threading
    enqueue_thread = threading.Thread(target=enqueue_captions)
    process_thread = threading.Thread(target=process_captions)
    save_thread = threading.Thread(target=save_results)
    
    # Start the pipeline
    enqueue_thread.start()
    process_thread.start()
    save_thread.start()
    
    # Wait for all threads to complete
    enqueue_thread.join()
    process_thread.join()
    save_thread.join()
    
    # Close the progress bar
    pbar.close()
    
    # Wait for all I/O operations to complete and check for errors
    for future in as_completed(io_futures):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error saving T5 embedding: {str(e)}")
    
    io_pool.shutdown()
    logger.info(f"Completed processing {total_samples} T5 embeddings")

if __name__ == "__main__":
    main() 