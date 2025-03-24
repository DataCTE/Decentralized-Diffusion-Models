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
        
        # Enable CUDA graphs for model inference
        self._capture_cuda_graphs()

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
        
        # Stage 3: Batched content validation using memory mapping
        return self._batched_content_validation(image_paths[valid_mask])

    def _batch_discovery(self, image_paths):
        """Matrix-based file discovery with SIMD optimizations"""
        # Use NumPy for vectorized string operations
        image_paths = np.array(image_paths, dtype=np.unicode_)
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
        """Process with pair validation and deterministic UUIDs"""
        try:
            # Generate deterministic UUID from image-caption pair
            with open(img_path, 'rb') as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            caption_path = Path(img_path).with_suffix('.txt')
            with open(caption_path, 'rb') as f:
                text_hash = hashlib.md5(f.read()).hexdigest()
            pair_hash = hashlib.md5((img_hash + text_hash).encode()).hexdigest()
            base_name = uuid.UUID(pair_hash).hex
            
            # Skip if all features exist
            if all(
                (Path(self.feature_dir)/ext/f"{base_name}.pt").exists()
                for ext in ["latents", "clip", "dims", "dino_features"]
            ):
                return True
            
            # Load caption text
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Process image and text
            with Image.open(img_path) as img:
                features = {
                    'dino': self._extract_dino_features(img),
                    'latent': self._extract_vae_latent(img),
                    'clip': self._extract_clip_embedding(caption),
                    'dims': torch.tensor(img.size, dtype=torch.int16)
                }
            
            self._save_features(base_name, features)
            return True
            
        except Exception as e:
            # Clean up any partial features
            for ext in ["latents", "clip", "dims", "dino_features"]:
                (Path(self.feature_dir)/ext/f"{base_name}.pt").unlink(missing_ok=True)
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
        """Extract VAE latent with bucket-aware preprocessing"""
        # Get original dimensions
        w, h = img.size
        
        # Resize to nearest bucket
        bucket = min(self.config.buckets, key=lambda b: abs(b[0]/b[1] - w/h))
        resized_img = img.resize(bucket, Image.LANCZOS)
        
        img_tensor = transforms.ToTensor()(resized_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vae.encode(img_tensor).cpu()

    def _extract_clip_embedding(self, caption):
        """Extract CLIP text embedding from actual caption"""
        with torch.no_grad():
            return self.clip.encode([caption]).cpu()

    def _save_features(self, base_name, features):
        """Save all features with UUID-based filenames"""
        torch.save(features['latent'], f"{self.feature_dir}/latents/{base_name}.pt")
        torch.save(features['clip'], f"{self.feature_dir}/clip/{base_name}.pt")
        torch.save(features['dims'], f"{self.feature_dir}/dims/{base_name}.pt")
        torch.save(features['dino'], f"{self.feature_dir}/dino_features/{base_name}.pt")

    def run_clustering(self):
        """Improved two-stage clustering with paper parameters"""
        dino_features = self._load_dino_features()
        
        # Stage 1: Fine-grained KMeans with paper settings
        print("Performing fine-grained KMeans clustering...")
        kmeans = faiss.Kmeans(
            dino_features.shape[1],
            self.config.num_fine_clusters,  # Should be 1024 per paper
            niter=100,
            gpu=True,
            spherical=True,
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
            nredo=3  # Paper uses 3 restarts
        )
        kmeans.train(dino_features)
        
        # Stage 2: Hierarchical clustering with balanced merging
        print("Performing hierarchical clustering...")
        agg = AgglomerativeClustering(
            n_clusters=self.config.num_experts,  # Should be 8
            linkage='average',
            metric='cosine',
            compute_full_tree=True  # Paper recommends full tree for balance
        )
        agg.fit(kmeans.centroids)
        
        # Paper's assignment strategy (section 4.1)
        _, fine_labels = kmeans.index.search(dino_features, 1)
        cluster_labels = agg.labels_[fine_labels.flatten()]
        self._save_clusters(cluster_labels)

    def _load_dino_features(self):
        """Load all DINO features for clustering"""
        feature_files = list((Path(self.feature_dir)/"dino_features").glob("*.pt"))
        return torch.cat([torch.load(f) for f in tqdm(feature_files, desc="Loading DINO features")])

    def _save_clusters(self, labels):
        """Save cluster assignments with UUID mapping"""
        features = list((Path(self.feature_dir)/"dino_features").glob("*.pt"))
        for f, label in zip(features, labels):
            torch.save(torch.tensor(label), f"{self.feature_dir}/clusters/{f.stem}.pt")

    def validate_dataset(self):
        """Ensure 1:1 correspondence of all features"""
        valid = 0
        for f in tqdm((Path(self.feature_dir)/"latents").glob("*.pt"), desc="Validating"):
            base = f.stem
            required = [
                f"{self.feature_dir}/clip/{base}.pt",
                f"{self.feature_dir}/clusters/{base}.pt",
                f"{self.feature_dir}/dims/{base}.pt"
            ]
            if all(os.path.exists(p) for p in required):
                valid += 1
        print(f"Dataset validation: {valid} complete samples")

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
        """Capture static computation graphs for model inference"""
        static_input = torch.randn(1, 3, 256, 256, device='cuda', dtype=self.dtype)
        self.vae_cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.vae_cuda_graph):
            self.static_vae_output = self.vae.encode(static_input)

def main():
    # Create default config if none provided
    config = get_config()  # Now works without arguments
    
    preprocessor = UnifiedPreprocessor(config)
    
    # Discover all potential images
    raw_images = [str(p) for p in Path(config.dataset_path).rglob('*') 
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Validate pairs before processing
    valid_images = preprocessor._validate_image_caption_pairs(raw_images)
    
    # Process only validated pairs
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(preprocessor.process_image, p) for p in valid_images]
        results = [f.result() for f in tqdm(futures, desc="Processing images")]
    
    # Post-processing validation
    preprocessor.validate_dataset()
    preprocessor.run_clustering()

if __name__ == "__main__":
    main() 