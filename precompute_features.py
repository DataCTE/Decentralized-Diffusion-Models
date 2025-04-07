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
import toml
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from types import SimpleNamespace
import fire
import logging
from data.clustering import DDMClustering
from utils.distributed import setup_distributed

# Import local modules (assuming correct paths relative to project root)
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from data.t5 import T5TextEncoder
from utils import dict_to_sns # Assuming dict_to_sns is in utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Dataset for Precomputation ---
class PrecomputeDataset(Dataset):
    """Basic dataset to load images and captions."""
    def __init__(self, config):
        self.root_dir = Path(config.dataset_path)
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
        self.transform = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS), # Example resize
            transforms.CenterCrop(512)
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
            # Apply minimal transform if needed, or let extractors handle it
            # img = self.transform(img) # Optional basic transform
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
        # PIL Images are passed; VAEWrapper or specific extraction logic handles format.
        # Assuming VAEWrapper handles PIL -> Tensor -> Normalize internally if needed,
        # otherwise, preprocessing is needed here. Let's assume VAEWrapper expects tensors.
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
        ])
        try:
            # Stack preprocessed images into a batch tensor
            # Filter out None images before preprocessing
            valid_images = [img for img in images_pil if img is not None]
            if not valid_images: return None # Skip if no valid images in batch
            img_tensors = torch.stack([preprocess(img) for img in valid_images]).to(self.device)

            with torch.no_grad():
                # Precision handled within VAEWrapper encode
                latents = self.vae.encode(img_tensors)
            # Handle potential mismatch if some images were None
            # Need to return a tensor of the original batch size with placeholders for errors
            # This is complex. Simplification: Assume no errors or handle in process_batch.
            # Returning potentially smaller tensor here. process_batch needs adjustment or error handling.
            # For now, returning the latents of valid images.
            return latents.cpu() # Return on CPU
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
            embed_dim = self.clip.model.config.hidden_size if hasattr(self.clip.model, 'config') else 768
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
        return torch.zeros(len(images_pil), 1024) # Placeholder

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
             logger.warning(f"[Rank {self.rank}] Skipping save for {feature_type} batch {batch_file_index} due to None data.")
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
        images_pil = batch_data['image'] # List of PIL images or Nones
        captions = batch_data['caption'] # List of strings
        # Filter out None images and corresponding captions
        valid_indices = [i for i, img in enumerate(images_pil) if img is not None]
        if len(valid_indices) != len(images_pil):
            logger.warning(f"[Rank {self.rank}] Batch {batch_file_index}: Found {len(images_pil) - len(valid_indices)} loading errors. Processing {len(valid_indices)} valid samples.")
            if not valid_indices: # Skip batch if all images failed
                 return
            images_pil = [images_pil[i] for i in valid_indices]
            captions = [captions[i] for i in valid_indices]

        batch_results = {}
        if 'dims' in self.enabled_features:
             batch_results['dims'] = self._extract_batch_dims(images_pil)
        if 'buckets' in self.enabled_features and 'dims' in batch_results:
             batch_results['buckets'] = self._extract_batch_buckets(batch_results['dims'])
        if 'vae' in self.enabled_features:
             batch_results['vae'] = self._extract_batch_vae(images_pil)
        if 'clip' in self.enabled_features:
             batch_results['clip'] = self._extract_batch_clip(captions)
        if 't5' in self.enabled_features:
             batch_results['t5'] = self._extract_batch_t5(captions)
        if 'dino' in self.enabled_features:
             batch_results['dino'] = self._extract_batch_dino(images_pil)

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
            # We can derive this using the dataloader's sampler current epoch and batch index if needed,
            # but simpler is to use rank and local batch_idx for the filename.
            # DDMDataset loading logic will need to handle discovering files like rank_batchidx.pt
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

        all_files = sorted([f for f in feature_subdir.glob(f"*.pt")]) # Find all .pt files

        # --- Subsampling ---
        num_files_to_sample = int(len(all_files) * subsample_fraction)
        if num_files_to_sample < 1 and len(all_files) > 0:
             num_files_to_sample = 1 # Sample at least one file if available
        if num_files_to_sample == 0:
             logger.warning("No feature files found or subsample fraction too small. Cannot cluster.")
             return None

        # Sample file indices globally consistently (same sample across all ranks if needed, though only rank 0 clusters)
        # Use a fixed seed for reproducibility if desired
        random.seed(42) # Use a fixed seed
        sampled_file_indices = random.sample(range(len(all_files)), num_files_to_sample)
        files_to_load_paths = [all_files[i] for i in sorted(sampled_file_indices)]

        logger.info(f"[Rank {self.rank}] Loading {len(files_to_load_paths)}/{len(all_files)} '{feature_type}' files for subsampled clustering.")

        # Load features (only rank 0 needs to do this for actual clustering)
        features_list = []
        if self.rank == 0:
            for file_path in tqdm(files_to_load_paths, desc="Loading subsampled features"):
                try:
                    # Load directly to CPU to manage memory
                    batch_features = torch.load(file_path, map_location='cpu')
                    features_list.append(batch_features.float()) # Ensure float32
                except Exception as e:
                    logger.error(f"Error loading feature file {file_path}: {e}")
            if not features_list:
                 logger.error("Failed to load any features for clustering.")
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
                # Raise an error or signal failure appropriately
                raise RuntimeError("Clustering feature loading failed.")

            # --- Use DDMClustering ---
            logger.info(f"Initializing DDMClustering with {num_coarse} coarse and {num_fine} fine clusters.")
            # Pass feature_path for potential internal saving/loading if DDMClustering needs it
            clustering_module = DDMClustering(num_coarse, num_fine, feature_path=self.feature_dir)

            # Perform clustering
            # Modify DDMClustering.cluster to accept features directly and return assignments
            # Assuming DDMClustering.cluster now takes features and returns assignments
            final_assignments = clustering_module.cluster(features=clustering_features) # Pass loaded features

            if final_assignments is None:
                 raise RuntimeError("Clustering module failed to return assignments.")

            # --- Save Assignments (Rank 0 handles this) ---
            logger.info(f"Clustering complete. Saving assignments for {len(final_assignments)} samples...")
            assignments_dir = self.feature_dir / "clusters"
            assignments_dir.mkdir(exist_ok=True)

            # The current implementation saves assignments sequentially based on the order
            # of 'dims' files. This works correctly ONLY IF subsample_fraction=1.0
            # AND the order of features loaded for clustering matches the order of dims files.
            # Using subsampling < 1.0 requires a more complex mapping strategy.
            if subsample_fraction < 1.0:
                 logger.warning("Saving cluster assignments with subsampling < 1.0. The current sequential saving relies on the order of processed 'dims' files matching the order of the (subsampled) features used for clustering. Verify this assumption holds for your subsampling method, otherwise assignments might be incorrect.")

            # Assume we clustered all features from rank 0's loaded files
            # We need to distribute these assignments back into the per-batch file structure.
            # Load the 'dims' files to know how many samples are in each original batch file.
            dims_dir = self.feature_dir / "dims"
            all_dims_files = sorted([f for f in dims_dir.glob(f"*.pt")]) # Load all dims files

            assignments_idx = 0
            total_samples_processed = 0
            for dims_file_path in tqdm(all_dims_files, desc="Saving cluster assignments per file"):
                try:
                    batch_dims = torch.load(dims_file_path, map_location='cpu')
                    num_samples_in_batch = batch_dims.shape[0]
                    
                    # Check if we have enough assignments left
                    if assignments_idx + num_samples_in_batch > len(final_assignments):
                        logger.error(f"Mismatch between number of clustered samples ({len(final_assignments)}) and total samples found in dims files. Stopping assignment saving.")
                        # This indicates an issue, possibly with subsampling or file loading.
                        break 

                    batch_assignments = final_assignments[assignments_idx : assignments_idx + num_samples_in_batch]

                    # Derive the cluster save path from the dims file path
                    # Assumes dims filename format is rank_batchidx.pt
                    base_filename = dims_file_path.name
                    cluster_filename = base_filename # Use the same rank_batchidx.pt format
                    cluster_save_path = assignments_dir / cluster_filename

                    torch.save(batch_assignments.cpu(), cluster_save_path)
                    assignments_idx += num_samples_in_batch
                    total_samples_processed += num_samples_in_batch

                except Exception as e:
                    logger.error(f"Error processing/saving assignments for file corresponding to {dims_file_path.name}: {e}")
                    # Decide whether to continue or stop

            if assignments_idx != len(final_assignments):
                 logger.warning(f"Processed assignments ({assignments_idx}) does not match total assignments ({len(final_assignments)}). There might be an issue.")
            logger.info(f"Finished assigning clusters to {total_samples_processed} samples.")
            
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
    # Example: Generate all standard features. Modify based on needs.
    # Clustering requires 'dims' and the feature specified in 'clustering_feature_type'.
    # DDMDataset requires 'latents', 'clip', 't5', 'dims', 'buckets', 'clusters'.
    features_to_generate = {'vae', 'clip', 't5', 'dims', 'buckets'}
    features_for_clustering = {cfg.data.clustering_feature_type, 'dims'} # Need dims to save assignments correctly
    features_needed = features_to_generate.union(features_for_clustering)
    logger.info(f"Enabled features for generation/checking: {features_needed}")

    # --- Initialize Generator ---
    generator = FeatureGenerator(cfg, enabled_features=features_needed)

    # --- Initialize Dataset & Dataloader ---
    logger.info("Initializing Dataset and DataLoader...")
    dataset = PrecomputeDataset(cfg) # Uses simple list of file paths

    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) # No shuffle for precompute
        # Use a larger batch size for precomputation if memory allows
        precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 128)
        dataloader = DataLoader(
            dataset,
            batch_size=precompute_batch_size, # Use dedicated precompute batch size
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=False # Process all samples
        )
    else:
        # Non-distributed dataloader
        precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 128)
        dataloader = DataLoader(
            dataset,
            batch_size=precompute_batch_size, # Use dedicated precompute batch size
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=False
        )
    logger.info(f"DataLoader initialized with batch size {precompute_batch_size}.")


    # --- Run Feature Extraction ---
    if not skip_feature_extraction:
        generator.run_feature_extraction(dataloader)
    else:
        logger.info("Skipping feature extraction.")
    # Add barrier after extraction or skip
    if distributed:
        logger.info(f"Rank {rank} waiting at barrier after feature extraction step.")
        dist.barrier()

    # --- Run Clustering ---
    # Only rank 0 performs clustering after all ranks finish extraction
    if not skip_clustering:
        generator.run_clustering() # run_clustering now handles the rank check and barriers internally
    else:
        logger.info("Skipping clustering.")
        # Barrier needed even if clustering skipped by rank 0, others wait
        if distributed:
             logger.info(f"Rank {rank} waiting at barrier after skipping clustering step.")
             dist.barrier()

    # --- Cleanup ---
    if distributed:
            dist.destroy_process_group()
    logger.info("Precomputation finished.")


if __name__ == "__main__":
    fire.Fire(main) 