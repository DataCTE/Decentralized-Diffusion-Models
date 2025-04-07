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
import toml
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from types import SimpleNamespace
import fire
import logging
from data.clustering import DDMClustering

# Import local modules (assuming correct paths relative to project root)
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from data.t5 import T5TextEncoder
from utils import dict_to_sns # Assuming dict_to_sns is in utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ADD setup_distributed function definition here ---
def setup_distributed():
    """Initializes torch.distributed"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Initializing distributed training: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        # Ensure backend is explicitly set if needed, nccl is common for NVIDIA GPUs
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
             torch.cuda.set_device(local_rank)
             device = torch.device(f"cuda:{local_rank}")
        else:
             device = torch.device("cpu") # Fallback for CPU-only distributed (less common)
        return rank, world_size, local_rank, device
    else:
        print("Not running in distributed mode.")
        # Setup for single GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device # rank, world_size, local_rank, device
# --- End of setup_distributed function definition ---

# --- Dataset for Precomputation ---
class PrecomputeDataset(Dataset):
    """Basic dataset to load images and captions."""
    def __init__(self, config):
        self.root_dir = Path(config.data.dataset_path)
        self.image_files = []
        self.captions = {} # Maps image filename (without ext) to caption

        logger.info(f"Scanning dataset at {self.root_dir}...")
        # Example: Assume images are in root_dir and captions in root_dir/captions.txt
        # Adjust this logic based on your actual dataset structure
        caption_file = self.root_dir / "captions.txt" # Example caption file path
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1) # Example: image_name.jpg\tCaption text
                    if len(parts) == 2:
                        img_name_no_ext = os.path.splitext(parts[0])[0]
                        self.captions[img_name_no_ext] = parts[1]
        except FileNotFoundError:
            logger.warning(f"Caption file not found at {caption_file}. Text features (CLIP/T5) cannot be generated.")
            self.captions = {}

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        for entry in os.scandir(self.root_dir):
            if entry.is_file() and entry.name.lower().endswith(valid_extensions):
                 self.image_files.append(entry.path)

        logger.info(f"Found {len(self.image_files)} images.")
        if not self.captions:
            logger.warning("No captions loaded. CLIP/T5 features will be skipped or use empty strings.")

        # Basic image transform: resize and convert to tensor
        # VAEWrapper expects [-1, 1] range, encoders expect specific formats
        # We load PIL images and let each extractor handle its required transform
        img_size = getattr(config.data, 'precompute_image_size', 512) # Add a config option if needed
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(img_size)
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_filename = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_filename)[0]

        try:
            # Load image as PIL
            img = Image.open(img_path).convert('RGB')
            # Apply transform here if decided it's needed before feature extraction
            # img = self.transform(img)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}. Returning None.")
            img = None # Signal error

        # Get caption, default to empty string if missing or no caption file
        caption = self.captions.get(img_name_no_ext, "")

        # Return raw PIL image, caption, and original path/ID
        return {'image': img, 'caption': caption, 'id': img_filename}

# --- Feature Generator ---
class FeatureGenerator:
    def __init__(self, config, enabled_features):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.distributed = False
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.distributed = True
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"[Rank {self.rank}] Initializing FeatureGenerator on device: {self.device}")
        self.enabled_features = set(enabled_features) # Use a set for faster checks
        self.feature_dir = Path(config.data.feature_cache_path)

        # Initialize models ONLY if the corresponding feature is enabled
        self.vae = None
        self.clip = None
        self.t5 = None
        self.dino = None

        # Pass relevant sub-config to wrappers/encoders
        data_cfg = config.data
        train_cfg = config.train # Need use_mixed_precision

        # Combine necessary fields for sub-configs
        vae_sub_config = SimpleNamespace(
             **{k: v for k, v in data_cfg.__dict__.items() if k.startswith('vae_')},
             use_mixed_precision=train_cfg.use_mixed_precision,
             latent_channels=data_cfg.latent_channels # Ensure latent_channels is included
        )
        # Ensure vae_model is present if vae is enabled
        if 'vae' in self.enabled_features and not hasattr(vae_sub_config, 'vae_model'):
             raise ValueError("VAE feature enabled, but 'vae_model' not found in [data] config.")

        clip_sub_config = SimpleNamespace(
             clip_model_name=data_cfg.clip_model_name,
             clip_max_token_length=getattr(data_cfg, 'clip_max_token_length', 77),
             use_mixed_precision=train_cfg.use_mixed_precision
        )
        t5_sub_config = SimpleNamespace(
             t5_model_name=data_cfg.t5_model_name,
             t5_max_token_length=getattr(data_cfg, 't5_max_token_length', 128),
             use_mixed_precision=train_cfg.use_mixed_precision
        )

        if 'vae' in self.enabled_features:
            logger.info(f"[Rank {self.rank}] Initializing VAE...")
            self.vae = VAEWrapper(self.device, vae_sub_config)
        if 'clip' in self.enabled_features:
            logger.info(f"[Rank {self.rank}] Initializing CLIP...")
            self.clip = CLIPTextEncoder(self.device, clip_sub_config)
        if 't5' in self.enabled_features:
            logger.info(f"[Rank {self.rank}] Initializing T5...")
            self.t5 = T5TextEncoder(self.device, t5_sub_config)
        if 'dino' in self.enabled_features:
            logger.info(f"[Rank {self.rank}] Initializing DINO...")
            # DINO feature extraction can be implemented here if needed.
            # self.dino = ... load DINO model ...
            logger.warning("DINO model loading/extraction not implemented. Skipping 'dino' feature.")
            self.enabled_features.discard('dino') # Disable if not loaded
        if 'dims' in self.enabled_features or 'buckets' in self.enabled_features:
             # Store bucketing config for direct use
             self.buckets = getattr(data_cfg, 'buckets', None)
             self.bucket_scale = getattr(data_cfg, 'bucket_scale', None)
             if not self.buckets or not self.bucket_scale:
                  logger.warning("Bucketing enabled but 'buckets' or 'bucket_scale' missing in config. Disabling.")
                  self.enabled_features.discard('buckets')

        # Create directories on rank 0
        self._create_dirs()

    def _create_dirs(self):
        """Create output directories for enabled features on rank 0."""
        if self.rank == 0:
            dir_map = {
                'vae': 'latents', 'clip': 'clip', 't5': 't5', 'dino': 'dino',
                'dims': 'dims', 'buckets': 'buckets', 'clusters': 'clusters'
            }
            self.feature_dir.mkdir(parents=True, exist_ok=True)
            for feat, dirname in dir_map.items():
                if feat in self.enabled_features:
                    (self.feature_dir / dirname).mkdir(exist_ok=True)
        if self.distributed:
            dist.barrier()

    # --- Batch Feature Extraction Methods ---
    def _extract_batch_vae(self, images_pil: list):
        """Encodes a batch of PIL images into latents."""
        if not self.vae: return None
        # VAEWrapper expects [-1, 1] normalized tensors.
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
        ])
        try:
            # Filter out None images before preprocessing
            valid_images = [img for img in images_pil if img is not None]
            if not valid_images: return None # Skip if no valid images in batch
            img_tensors = torch.stack([preprocess(img) for img in valid_images]).to(self.device)

            with torch.no_grad():
                # Precision handled within VAEWrapper encode
                latents = self.vae.encode(img_tensors)

            # Need to handle batches where some images failed to load
            # Create a full-size placeholder and fill valid latents
            full_latents = None
            if len(valid_images) < len(images_pil):
                 # Determine latent shape from the first valid latent
                 if latents.shape[0] > 0:
                      latent_c, latent_h, latent_w = latents.shape[1:]
                      full_latents = torch.zeros(len(images_pil), latent_c, latent_h, latent_w, dtype=latents.dtype)
                      valid_indices_iter = iter(range(len(valid_images)))
                      for i in range(len(images_pil)):
                           if images_pil[i] is not None:
                                valid_idx = next(valid_indices_iter)
                                full_latents[i] = latents[valid_idx]
                      # else: latents remain zeros (placeholder for failed image)
                 else: # All images failed to preprocess/encode? Return None
                      return None
            else: # All images were valid
                 full_latents = latents

            return full_latents.cpu() # Return on CPU
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Error in VAE batch encoding: {e}")
            # Return None to indicate batch failure
            return None


    def _extract_batch_clip(self, captions: list):
        """Encodes a batch of captions using CLIP pooler."""
        if not self.clip: return None
        try:
            with torch.no_grad(): # Encoder handles internal context
                pooled_embeddings = self.clip.encode_pooled(captions)
            return pooled_embeddings.cpu() # Return on CPU
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Error in CLIP batch encoding: {e}")
            # Attempt to get embed_dim more robustly
            try:
                 embed_dim = self.clip.model.text_projection.shape[-1]
            except:
                 embed_dim = 768 # Fallback
            return torch.zeros(len(captions), embed_dim) # Placeholder

    def _extract_batch_t5(self, captions: list):
        """Encodes a batch of captions using T5 sequence."""
        if not self.t5: return None
        try:
            with torch.no_grad(): # Encoder handles internal context
                sequence_embeddings = self.t5.encode(captions)
            return sequence_embeddings.cpu() # Return on CPU
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Error in T5 batch encoding: {e}")
            hidden_size = getattr(self.t5.model.config, 'd_model', 1024)
            max_len = self.t5.max_length
            return torch.zeros(len(captions), max_len, hidden_size) # Placeholder

    def _extract_batch_dino(self, images_pil: list):
        """Encodes a batch of PIL images using DINO."""
        # Placeholder - Implement DINO batch processing if needed
        logger.warning("Batch DINO extraction not implemented.")
        if not self.dino: return None
        # Filter out None images
        valid_images = [img for img in images_pil if img is not None]
        if not valid_images: return torch.zeros(len(images_pil), 1024) # Return placeholder if all failed
        # Process valid images (placeholder)
        num_valid = len(valid_images)
        # Assume DINO output dim is 1024
        dino_embeddings = torch.rand(num_valid, 1024) # Replace with actual DINO processing

        # Create full batch tensor with placeholders for failed images
        full_embeddings = torch.zeros(len(images_pil), 1024)
        valid_indices_iter = iter(range(num_valid))
        for i in range(len(images_pil)):
            if images_pil[i] is not None:
                valid_idx = next(valid_indices_iter)
                full_embeddings[i] = dino_embeddings[valid_idx]
        return full_embeddings

    def _extract_batch_dims(self, images_pil: list):
        """Extracts dimensions for a batch of PIL images."""
        dims = []
        for img in images_pil:
            if img:
                dims.append(list(img.size)) # [width, height]
            else:
                dims.append([0, 0]) # Placeholder for errors
        return torch.tensor(dims, dtype=torch.int16) # B x 2

    def _get_bucket_index(self, width, height):
        """Finds the bucket index for a given dimension."""
        # Simple Manhattan distance to find closest bucket
        if not self.buckets: return 0 # Default to bucket 0 if not configured
        if width == 0 or height == 0: return 0 # Handle error case from dims
        target_w = round(width / self.bucket_scale) * self.bucket_scale
        target_h = round(height / self.bucket_scale) * self.bucket_scale
        min_dist = float('inf')
        best_idx = 0
        for idx, (bw, bh) in enumerate(self.buckets):
            dist = abs(target_w - bw) + abs(target_h - bh)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        return best_idx

    def _extract_batch_buckets(self, batch_dims: torch.Tensor):
        """Determines bucket indices for a batch based on dimensions."""
        if not self.buckets: return None
        bucket_indices = [self._get_bucket_index(w, h) for w, h in batch_dims.tolist()]
        return torch.tensor(bucket_indices, dtype=torch.int16) # B

    def _save_batch_feature(self, feature_type: str, batch_data: torch.Tensor, batch_file_index: int):
        """Saves a batch tensor for a specific feature type."""
        if batch_data is None:
             # Don't log a warning here, None indicates feature was disabled or failed earlier (logged there)
             return

        dir_map = {'vae': 'latents', 'clip': 'clip', 't5': 't5', 'dino': 'dino',
                   'dims': 'dims', 'buckets': 'buckets', 'clusters': 'clusters'}
        save_dir = self.feature_dir / dir_map[feature_type]
        # Ensure filename includes rank to avoid collisions in shared FS when not using DS sampler offset
        filename = f"{self.rank:03d}_{batch_file_index:06d}.pt"
        filepath = save_dir / filename
        try:
            torch.save(batch_data.cpu(), filepath) # Save CPU tensor
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Error saving {feature_type} batch to {filepath}: {e}")


    def process_batch(self, batch_data: dict, batch_file_index: int):
        """Processes a batch: extracts enabled features and saves them."""
        # Note: batch_data['image'] contains PIL images OR Nones if loading failed
        images_pil = batch_data['image']
        captions = batch_data['caption']

        # Process dimensions and buckets first (works even with Nones in images_pil)
        batch_results = {}
        if 'dims' in self.enabled_features:
             batch_results['dims'] = self._extract_batch_dims(images_pil)
        if 'buckets' in self.enabled_features and 'dims' in batch_results:
             batch_results['buckets'] = self._extract_batch_buckets(batch_results['dims'])

        # Process features requiring valid images/captions
        if 'vae' in self.enabled_features:
             batch_results['vae'] = self._extract_batch_vae(images_pil) # Handles Nones internally
        if 'clip' in self.enabled_features:
             batch_results['clip'] = self._extract_batch_clip(captions) # Assumes captions exist even if image failed
        if 't5' in self.enabled_features:
             batch_results['t5'] = self._extract_batch_t5(captions)
        if 'dino' in self.enabled_features:
             batch_results['dino'] = self._extract_batch_dino(images_pil) # Handles Nones internally

        # Save each computed feature tensor
        for feature_type, batch_tensor in batch_results.items():
             self._save_batch_feature(feature_type, batch_tensor, batch_file_index)

    def run_feature_extraction(self, dataloader: DataLoader):
        """Runs feature extraction over the dataset using the dataloader."""
        logger.info(f"[Rank {self.rank}] Starting feature extraction...")
        start_time = time.time()

        # Each rank processes its slice of the data provided by the DistributedSampler
        # The batch_idx here is local to the rank's dataloader iterator
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Rank {self.rank} Processing", disable=(self.rank != 0))):
            # The batch_file_index needs to be globally unique across ranks.
            # We use rank and local batch_idx for the filename.
            # DDMDataset loading logic handles discovering files like rank_batchidx.pt
            self.process_batch(batch_data, batch_idx)

        if self.distributed:
            dist.barrier() # Wait for all ranks to finish saving

        end_time = time.time()
        logger.info(f"[Rank {self.rank}] Feature extraction finished in {end_time - start_time:.2f} seconds.")

    # --- Clustering Logic ---
    def _load_features_for_clustering(self, feature_type: str, subsample_fraction: float):
        """Loads features for clustering, potentially subsampled."""
        logger.info(f"[Rank {self.rank}] Loading '{feature_type}' features for clustering (subsample: {subsample_fraction})...")
        feature_subdir = self.feature_dir / feature_type
        if not feature_subdir.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_subdir}")

        # Find all files (rank_batchidx.pt) in the feature directory
        all_files = sorted(list(feature_subdir.glob("*_*.pt")))

        if not all_files:
             logger.warning(f"No feature files found in {feature_subdir}. Cannot cluster.")
             return None

        # --- Subsampling ---
        num_files_to_sample = int(len(all_files) * subsample_fraction)
        if num_files_to_sample < 1 :
             num_files_to_sample = 1 # Sample at least one file if available

        # Sample file indices globally consistently (same sample across all ranks if needed, though only rank 0 clusters)
        # Use a fixed seed for reproducibility if desired
        rng_state = random.getstate() # Store current RNG state
        random.seed(42) # Use a fixed seed for sampling files
        sampled_file_indices = random.sample(range(len(all_files)), num_files_to_sample)
        random.setstate(rng_state) # Restore original RNG state
        files_to_load_paths = [all_files[i] for i in sorted(sampled_file_indices)]

        logger.info(f"[Rank {self.rank}] Will load {len(files_to_load_paths)}/{len(all_files)} '{feature_type}' files for subsampled clustering.")

        # Load features (only rank 0 needs to do this for actual clustering)
        features_list = []
        original_indices_map = [] # To map subsampled feature index back to original file/index
        if self.rank == 0:
            current_original_idx = 0
            for file_path in tqdm(files_to_load_paths, desc="Loading subsampled features"):
                try:
                    # Load directly to CPU to manage memory
                    batch_features = torch.load(file_path, map_location='cpu')
                    # Check for NaNs/Infs in loaded features
                    if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                         logger.warning(f"NaN/Inf found in feature file {file_path}. Skipping this batch for clustering.")
                         continue
                    features_list.append(batch_features.float()) # Ensure float32

                    # Keep track of original indices if needed for assignment later (complex)
                    # For now, we assume assignments are saved sequentially based on *all* files
                    # num_in_batch = batch_features.shape[0]
                    # original_indices_map.extend(range(current_original_idx, current_original_idx + num_in_batch))
                    # current_original_idx += num_in_batch

                except Exception as e:
                    logger.error(f"Error loading feature file {file_path}: {e}")

            if not features_list:
                 logger.error("Failed to load any valid features for clustering.")
                 return None
            all_features = torch.cat(features_list, dim=0)
            logger.info(f"[Rank 0] Loaded subsampled features shape: {all_features.shape}")
            return all_features
        else:
            return None # Other ranks don't need the features for clustering


    def run_clustering(self):
        """Performs two-stage clustering and saves assignments."""
        if self.rank != 0:
            # Clustering is only done by rank 0
            if self.distributed:
                 logger.info(f"Rank {self.rank} waiting at barrier before clustering.")
                 dist.barrier()
                 logger.info(f"Rank {self.rank} waiting at barrier after clustering attempt.")
                 dist.barrier()
            return

        logger.info("Rank 0 starting clustering process...")
        try:
            feature_type = self.config.data.clustering_feature_type
            subsample_fraction = self.config.data.clustering_subsample_fraction
            num_coarse = self.config.model.num_clusters
            num_fine = self.config.data.num_fine_clusters

            # Load features for clustering (subsampled)
            clustering_features = self._load_features_for_clustering(feature_type, subsample_fraction)
            if clustering_features is None:
                logger.error("Failed to load features for clustering. Aborting.")
                raise RuntimeError("Clustering feature loading failed.")

            # --- Use DDMClustering ---
            logger.info(f"Initializing DDMClustering with {num_coarse} coarse and {num_fine} fine clusters.")
            clustering_module = DDMClustering(num_coarse, num_fine, feature_path=self.feature_dir)

            # Perform clustering on the loaded (potentially subsampled) features
            subsampled_assignments = clustering_module.cluster(features=clustering_features)

            if subsampled_assignments is None:
                 raise RuntimeError("Clustering module failed to return assignments.")

            # --- Assign ALL samples to clusters and Save Assignments ---
            logger.info("Clustering complete. Assigning all samples and saving assignments...")
            assignments_dir = self.feature_dir / "clusters"
            assignments_dir.mkdir(exist_ok=True)

            # Load ALL features of the specified type (only paths needed)
            feature_subdir = self.feature_dir / feature_type
            all_feature_files = sorted([f for f in feature_subdir.glob("*.pt")])

            if not all_feature_files:
                 raise FileNotFoundError(f"No feature files found in {feature_subdir} to assign clusters.")

            # Re-use the trained clustering module (centroids are stored) for assignment
            # Need Faiss index from DDMClustering if it wasn't saved/returned
            # Let's modify DDMClustering to keep the index accessible or re-create it
            # Assuming we can access/recreate the fine index from clustering_module:
            if clustering_module.fine_centroids is None or clustering_module.coarse_labels_for_fine is None:
                 raise RuntimeError("Clustering module did not retain necessary centroids/labels for assignment.")

            # Create Faiss index for fine centroids (Inner Product for cosine sim)
            index_fine = faiss.IndexFlatIP(clustering_module.fine_centroids.shape[1])
            res = None
            if clustering_module.use_gpu:
                res = faiss.StandardGpuResources()
                index_fine = faiss.index_cpu_to_gpu(res, 0, index_fine)
            index_fine.add(np.ascontiguousarray(clustering_module.fine_centroids, dtype=np.float32))
            coarse_labels_for_fine = clustering_module.coarse_labels_for_fine # Get the mapping

            total_samples_processed = 0
            for file_path in tqdm(all_feature_files, desc="Assigning clusters to all samples"):
                try:
                    batch_features = torch.load(file_path, map_location='cpu').float().numpy()
                    batch_features_np = np.ascontiguousarray(batch_features, dtype=np.float32)

                    # Search for nearest fine centroid for each feature in the batch
                    _, fine_centroid_indices = index_fine.search(batch_features_np, 1)
                    fine_centroid_indices = fine_centroid_indices.squeeze() # Shape (N_batch,)

                    # Map fine centroid indices to coarse labels
                    batch_assignments = coarse_labels_for_fine[fine_centroid_indices] # Shape (N_batch,)

                    # Derive the cluster save path from the feature file path
                    base_filename = file_path.name
                    cluster_filename = base_filename # Use the same rank_batchidx.pt format
                    cluster_save_path = assignments_dir / cluster_filename

                    torch.save(batch_assignments.cpu().short(), cluster_save_path) # Save as short tensor
                    total_samples_processed += batch_features.shape[0]

                except Exception as e:
                    logger.error(f"Error processing/assigning assignments for file {file_path.name}: {e}")

            logger.info(f"Finished assigning clusters to {total_samples_processed} samples across all files.")

        except Exception as e:
            logger.exception(f"Clustering failed: {e}") # Log full traceback
        finally:
            # Ensure barrier happens even if rank 0 fails
            if self.distributed:
                logger.info("Rank 0 waiting at barrier after clustering attempt.")
                dist.barrier()


# --- Main Execution ---
def main(config_path: str = "config.toml",
         skip_feature_extraction: bool = False,
         skip_clustering: bool = False):
    """Main function to run feature precomputation and clustering."""
    # --- Setup Distributed ---
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    distributed = world_size > 1

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    if not is_main: # Reduce logging noise from non-main ranks
        logger.setLevel(logging.WARNING)


    # --- Load Config ---
    try:
        config_dict = toml.load(config_path)
        cfg = dict_to_sns(config_dict) # Convert to SimpleNamespace
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return # Exit if config fails

    # --- Define Features to Generate ---
    # Clustering requires 'dims' and the feature specified in 'clustering_feature_type'.
    # DDMDataset requires 'latents', 'clip', 't5', 'dims', 'buckets', 'clusters'.
    features_to_generate = {'vae', 'clip', 't5', 'dims', 'buckets'}
    features_for_clustering = {cfg.data.clustering_feature_type, 'dims'} # Need dims to save assignments correctly
    features_needed = features_to_generate.union(features_for_clustering)
    if 'dino' in getattr(cfg.data, 'enabled_features', []): # Check if DINO is explicitly enabled
         features_needed.add('dino')
    logger.info(f"Enabled features for generation/checking: {features_needed}")

    # --- Initialize Generator ---
    generator = FeatureGenerator(cfg, enabled_features=features_needed)

    # --- Initialize Dataset & Dataloader ---
    logger.info("Initializing Dataset and DataLoader...")
    dataset = PrecomputeDataset(cfg) # Pass the namespace config

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) # No shuffle for precompute
        precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 128)
        dataloader = DataLoader(
            dataset,
            batch_size=precompute_batch_size, # Use dedicated precompute batch size
            sampler=sampler,
            num_workers=getattr(cfg.train, 'num_workers', 4), # Get num_workers from train config
            pin_memory=True,
            drop_last=False # Process all samples
        )
    else:
        precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 128)
        dataloader = DataLoader(
            dataset,
            batch_size=precompute_batch_size, # Use dedicated precompute batch size
            shuffle=False,
            num_workers=getattr(cfg.train, 'num_workers', 4), # Get num_workers from train config
            pin_memory=True,
            drop_last=False
        )
    logger.info(f"DataLoader initialized with batch size {precompute_batch_size}.")


    # --- Run Feature Extraction ---
    if not skip_feature_extraction:
        generator.run_feature_extraction(dataloader)
    else:
        logger.info("Skipping feature extraction.")
    if distributed:
        logger.info(f"Rank {rank} waiting at barrier after feature extraction step.")
        dist.barrier()

    # --- Run Clustering ---
    # Only rank 0 performs clustering after all ranks finish extraction
    if not skip_clustering:
        generator.run_clustering() # run_clustering now handles the rank check and barriers internally
    else:
        logger.info("Skipping clustering.")
        if distributed:
             logger.info(f"Rank {rank} waiting at barrier after skipping clustering step.")
             dist.barrier()

    # --- Cleanup ---
    if distributed:
            dist.destroy_process_group()
    logger.info("Precomputation finished.")


if __name__ == "__main__":
    fire.Fire(main) 