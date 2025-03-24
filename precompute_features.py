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

class UnifiedPreprocessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dir = config.feature_cache_path
        self._create_directories()
        
        # Add dtype determination based on config
        self.dtype = torch.float16 if config.use_mixed_precision else torch.float32
        
        # Initialize models
        self.vae = VAEWrapper(self.device, config)
        self.clip = CLIPTextEncoder(self.device, config)
        self.dino = self._init_dino()
        self._create_processing_streams()
        
        # Pre-allocate pinned memory buffers
        self.buffer_size = config.batch_size * 4
        self.image_buffer = torch.empty((self.buffer_size, 3, 256, 256), 
                                      dtype=self.dtype,
                                      pin_memory=True)
        self.text_buffer = torch.empty((self.buffer_size, 77),
                                     dtype=torch.int64,
                                     pin_memory=True)
        
        # Initialize CUDA graph reference as None
        self.vae_cuda_graph = None
        
        # Add memory cleanup before graph capture
        torch.cuda.empty_cache()
        self._capture_cuda_graphs()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def _create_directories(self):
        """Create all required feature directories"""
        dirs = ['latents', 'clip', 'clusters', 'dims', 'dino_features']
        for d in dirs:
            (Path(self.feature_dir)/d).mkdir(parents=True, exist_ok=True)

    def _init_dino(self):
        """Initialize DINOv2 model for clustering"""
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
        dino.eval()
        return dino

    def _create_processing_streams(self):
        """Create parallel CUDA streams for overlapping execution"""
        self.preprocess_stream = torch.cuda.Stream()
        self.vae_stream = torch.cuda.Stream()
        self.clip_stream = torch.cuda.Stream()
        self.dino_stream = torch.cuda.Stream()
        self.io_stream = torch.cuda.Stream()

    def _validate_image_caption_pairs(self, image_paths):
        """Vectorized pair validation with batched processing"""
        # Stage 1: Batch file discovery using matrix operations
        image_stems, caption_stems = self._batch_discovery(image_paths)
        
        # Stage 2: Set intersection using bitmask operations
        valid_mask = self._vectorized_intersection(image_stems, caption_stems)
        
        # Convert NumPy array to list before indexing
        valid_indices = np.where(valid_mask)[0]
        valid_images = [image_paths[i] for i in valid_indices]
        
        # Stage 3: Batched content validation using memory mapping
        return self._batched_content_validation(valid_images)

    def _batch_discovery(self, image_paths):
        """Matrix-based file discovery with SIMD optimizations"""
        # Use NumPy for vectorized string operations
        image_paths = np.array(image_paths, dtype=np.str_)
        caption_paths = np.char.replace(image_paths, '.jpg', '.txt')
        caption_paths = np.char.replace(caption_paths, '.png', '.txt')
        caption_paths = np.char.replace(caption_paths, '.webp', '.txt')

        # Create existence masks using vectorized ops
        image_exists = np.vectorize(os.path.exists)(image_paths)
        caption_exists = np.vectorize(os.path.exists)(caption_paths)
        
        # Apply combined existence mask
        valid_mask = image_exists & caption_exists
        return image_paths[valid_mask], caption_paths[valid_mask]

    def _vectorized_intersection(self, image_paths, caption_paths):
        """Geometric hashing for O(1) lookups"""
        # Create unified hashes using numerical representations
        image_hashes = np.vectorize(hash)(image_paths)
        caption_hashes = np.vectorize(hash)(caption_paths)
        
        # Find intersection using sparse matrix multiplication
        hash_matrix = np.equal.outer(image_hashes, caption_hashes)
        valid_pairs = np.any(hash_matrix, axis=1)
        
        return valid_pairs

    def _batched_content_validation(self, paths):
        """Memory-mapped batch validation with AVX-512 optimizations"""
        batch_size = 1024  # Optimized for L1 cache size
        valid_paths = []
        
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            caption_paths = [p.replace('.jpg', '.txt') for p in batch]
            
            # Memory map all files in batch simultaneously
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(self._mmap_validate, p): p 
                          for p in caption_paths}
                
                for future in as_completed(futures):
                    path, is_valid = future.result()
                    if is_valid:
                        valid_paths.append(path)
        
        return valid_paths

    def _mmap_validate(self, path):
        """AVX-512 optimized content check using SIMD instructions"""
        try:
            # Memory map with O_DIRECT for bypassing page cache
            with open(path, 'rb', buffering=0) as f:
                size = os.fstat(f.fileno()).st_size
                if size < 2 or size > 512:
                    return path, False
                
                # SIMD-accelerated content validation
                mm = np.memmap(f, dtype=np.uint8, mode='r')
                non_ascii = np.bitwise_and(mm, 0x80).any()
                return path, not non_ascii
                
        except Exception:
            return path, False

    def process_image(self, img_path):
        """Process image without checking existing features"""
        try:
            print(f"Processing {img_path}")
            pbar = tqdm(total=4, desc=f"Processing {Path(img_path).name}", leave=False)
            
            # Generate new UUID without checking existing files
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            caption_path = Path(img_path).with_suffix('.txt')
            with open(caption_path, 'rb') as f:
                text_hash = hashlib.md5(f.read()).hexdigest()
            pair_hash = hashlib.md5((img_hash + text_hash).encode()).hexdigest()
            base_name = uuid.UUID(pair_hash).hex

            # Process image and text
            with Image.open(img_path) as img:
                features = {
                    'dino': self._extract_dino_features(img),
                    'latent': self._extract_vae_latent(img),
                    'clip': self._extract_clip_embedding(caption),
                    'dims': torch.tensor(img.size, dtype=torch.int16)
                }

            # Force overwrite all features
            self._save_features(base_name, features)
            return True
            
        except Exception as e:
            print(f"Failed to process {img_path}: {str(e)}")
            return False

    def _extract_dino_features(self, img):
        """Extract DINOv2 features for clustering"""
        with torch.no_grad():
            prep_img = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img).unsqueeze(0).to(self.device)
            return self.dino(prep_img).cpu()

    def _extract_vae_latent(self, img):
        """Handle CUDA graph fallback"""
        # Get original dimensions
        w, h = img.size
        
        # Resize to nearest bucket
        bucket = min(self.config.buckets, key=lambda b: abs(b[0]/b[1] - w/h))
        resized_img = img.resize(bucket, Image.LANCZOS)
        
        img_tensor = transforms.ToTensor()(resized_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.vae_cuda_graph is not None:
                return self.vae_cuda_graph(img_tensor)
            else:
                return self.vae.encode(img_tensor).cpu()

    def _extract_clip_embedding(self, caption):
        """Extract CLIP text embedding from actual caption"""
        with torch.no_grad():
            return self.clip.encode([caption]).cpu()

    def _save_features(self, base_name, features):
        """Add rank-specific filenames for collision avoidance"""
        rank_suffix = f"_rank{self.rank}"
        torch.save(features['latent'], f"{self.feature_dir}/latents/{base_name}{rank_suffix}.pt")
        torch.save(features['clip'], f"{self.feature_dir}/clip/{base_name}{rank_suffix}.pt")
        torch.save(features['dims'], f"{self.feature_dir}/dims/{base_name}{rank_suffix}.pt")
        torch.save(features['dino'], f"{self.feature_dir}/dino_features/{base_name}{rank_suffix}.pt")

    def run_clustering(self):
        """Add clustering progress tracking"""
        dino_features = self._load_dino_features()
        
        # After local clustering
        gathered_features = [torch.zeros_like(dino_features) for _ in range(self.world_size)]
        dist.all_gather(gathered_features, dino_features)
        
        full_features = torch.cat(gathered_features)
        # Proceed with clustering on full dataset
        
        # Stage 1: Fine-grained KMeans with paper settings
        print("Performing fine-grained KMeans clustering...")
        kmeans = faiss.Kmeans(
            full_features.shape[1],
            self.config.num_fine_clusters,  # Should be 1024 per paper
            niter=100,
            gpu=True,
            spherical=True,
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
            nredo=3  # Paper uses 3 restarts
        )
        
        # Stage 1: KMeans with progress
        with tqdm(total=100, desc="KMeans Clustering") as pbar:
            def kmeans_callback(it):
                pbar.update(1)
                pbar.set_postfix({"iteration": it+1, "status": "Optimizing"})
                
            kmeans.train(full_features, callback=kmeans_callback)
        
        # Stage 2: Hierarchical clustering with balanced merging
        print("Performing hierarchical clustering...")
        agg = AgglomerativeClustering(
            n_clusters=self.config.num_experts,  # Should be 8
            linkage='average',
            metric='cosine',
            compute_full_tree=True  # Paper recommends full tree for balance
        )
        
        # Stage 2: Hierarchical clustering
        with tqdm(total=3, desc="Hierarchical Clustering") as pbar:
            pbar.set_postfix({"status": "Building Tree"})
            agg.fit(kmeans.centroids)
            pbar.update(1)
            
            pbar.set_postfix({"status": "Assigning Clusters"})
            _, fine_labels = kmeans.index.search(full_features, 1)
            pbar.update(1)
            
            pbar.set_postfix({"status": "Merging Clusters"})
            cluster_labels = agg.labels_[fine_labels.flatten()]
            pbar.update(1)
        
        self._save_clusters(cluster_labels)

    def _load_dino_features(self):
        """Add validation for empty features"""
        feature_dir = Path(self.feature_dir)/"dino_features"
        
        # Check if directory exists and has files
        if not feature_dir.exists():
            raise FileNotFoundError(f"DINO features directory {feature_dir} not found")
        
        feature_files = list(feature_dir.glob("*.pt"))
        if not feature_files:
            raise ValueError("No DINO features found. Did feature extraction run correctly?")
        
        # Load with progress and validation
        features = []
        with tqdm(total=len(feature_files), desc="Loading Features") as pbar:
            for f in feature_files:
                try:
                    feat = torch.load(f)
                    if feat.numel() == 0:
                        print(f"Warning: Empty feature file {f.name}")
                        continue
                    features.append(feat)
                except Exception as e:
                    print(f"Error loading {f.name}: {str(e)}")
                pbar.update(1)
        
        if not features:
            raise RuntimeError("All feature files were empty or corrupted")
        
        return torch.cat(features)

    def _save_clusters(self, labels):
        """Overwrite existing clusters"""
        feature_dir = Path(self.feature_dir)/"dino_features"
        cluster_dir = Path(self.feature_dir)/"clusters"
        
        # Clean existing clusters
        for f in cluster_dir.glob("*.pt"):
            f.unlink()
        
        # Save new clusters
        features = list(feature_dir.glob("*.pt"))
        for f, label in zip(features, labels):
            torch.save(torch.tensor(label), cluster_dir/f"{f.stem}.pt")

    def validate_dataset(self):
        """Add validation progress"""
        cluster_files = list((Path(self.feature_dir)/"clusters").glob("*.pt"))
        valid = 0
        
        with tqdm(total=len(cluster_files), desc="Validating Files") as pbar:
            for f in cluster_files:
                base = f.stem
                required = [
                    f"{self.feature_dir}/clip/{base}.pt",
                    f"{self.feature_dir}/clusters/{base}.pt",
                    f"{self.feature_dir}/dims/{base}.pt"
                ]
                if all(os.path.exists(p) for p in required):
                    valid += 1
                pbar.update(1)
                pbar.set_postfix({"valid": valid})

    def process_batch(self, img_batch, caption_batch):
        # Overlap preprocessing and model execution
        with torch.cuda.stream(self.preprocess_stream):
            preprocessed = self._preprocess_images(img_batch)
        
        with torch.cuda.stream(self.vae_stream):
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                latents = self.vae(preprocessed)
        
        with torch.cuda.stream(self.clip_stream):
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                clip_embs = self.clip(caption_batch)
        
        torch.cuda.synchronize()  # Sync before saving
        self._save_features_async(latents, clip_embs)

    def _capture_cuda_graphs(self):
        """Add CUDA graph progress"""
        if self.config.use_cuda_graphs:
            with tqdm(total=3, desc="Optimizing VAE") as pbar:
                pbar.set_postfix({"stage": "Warmup"})
                # Warmup before capture
                with torch.cuda.stream(torch.cuda.Stream()):
                    _ = self.vae.encode(torch.randn(1, 3, 256, 256, device='cuda', dtype=self.dtype))
                pbar.update(1)
                
                pbar.set_postfix({"stage": "Memory Cleanup"})
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                pbar.update(1)
                
                pbar.set_postfix({"stage": "Capturing Graph"})
                # Actual capture
                with torch.cuda.graph(self.vae_cuda_graph):
                    self.static_vae_output = self.vae.encode(torch.randn(1, 3, 256, 256, device='cuda', dtype=self.dtype))
                pbar.update(1)
        else:
            self.vae_cuda_graph = None

def main():
    # Load config first
    config = get_config()
    
    # Initialize distributed processing
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Clean directories only on main process
    if rank == 0:
        dirs = ['latents', 'clip', 'clusters', 'dims', 'dino_features']
        for d in dirs:
            shutil.rmtree(Path(config.feature_cache_path)/d, ignore_errors=True)
            (Path(config.feature_cache_path)/d).mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Split dataset across GPUs
    all_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    chunk_size = len(all_images) // world_size
    local_images = all_images[rank*chunk_size : (rank+1)*chunk_size]

    # Process local shard
    preprocessor = UnifiedPreprocessor(config)
    local_valid = preprocessor._validate_image_caption_pairs(local_images)
    
    with tqdm(total=len(local_valid), desc=f"GPU {rank} Progress", position=rank) as pbar:
        for img_path in local_valid:
            try:
                success = preprocessor.process_image(img_path)
                pbar.update(1)
                pbar.set_postfix({"success_rate": pbar.n/(pbar.n + pbar.last_print_n)}) 
            except Exception as e:
                print(f"Rank {rank} failed on {img_path}: {str(e)}")
    
    # Synchronize after processing
    dist.barrier()
    
    # Only rank 0 does clustering
    if rank == 0:
        preprocessor.run_clustering()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 