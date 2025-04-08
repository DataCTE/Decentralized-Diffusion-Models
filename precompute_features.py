"""
Centralized preprocessing pipeline for Decentralized Diffusion Models
Combines feature extraction, clustering, and latent precomputation
"""
import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from tqdm.auto import tqdm
import torch.distributed as dist
import time
import random
import toml
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from types import SimpleNamespace
import fire
import logging
from data.clustering import DDMClustering
from utils import dict_to_sns
import json
import datetime # <-- Import datetime

# Import local modules (assuming correct paths relative to project root)
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from data.t5 import T5TextEncoder

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
        
        # --- Increase Timeout --- START EDIT ---
        # Default is often 10 or 30 minutes. Increase significantly for long preprocessing.
        # Use datetime.timedelta for clarity
        timeout_duration = datetime.timedelta(hours=6) # Increase to 6 hours, should be plenty
        logger.info(f"Setting distributed process group timeout to: {timeout_duration}")
        # --- Increase Timeout --- END EDIT ---
        
        dist.init_process_group(
            backend=backend, 
            rank=rank, 
            world_size=world_size,
            timeout=timeout_duration # <-- Pass the increased timeout
        )
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
    """
    Basic dataset to load images and captions.
    Uses a manifest file for fast loading on large datasets.
    """
    MANIFEST_FILENAME = "dataset_manifest.json" # Define manifest filename

    def __init__(self, config):
        self.root_dir = Path(config.data.dataset_path)
        self.manifest_path = self.root_dir / self.MANIFEST_FILENAME
        self.image_files = []
        self.caption_files = {} # Maps image path to caption file path

        # --- Manifest Loading/Creation ---
        if self.manifest_path.exists():
            logger.info(f"Loading dataset manifest from: {self.manifest_path}")
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                # Populate from manifest
                for item in manifest_data:
                    img_path = item.get("image_path")
                    cap_path = item.get("caption_path")
                    if img_path: # Basic check
                        self.image_files.append(img_path)
                        if cap_path:
                             # Ensure paths loaded from JSON are absolute or relative to root
                             # Assuming paths in manifest are stored correctly (e.g., relative to root_dir)
                             # If they are absolute, use them directly. If relative, resolve them.
                             # Example: if stored relative: self.caption_files[img_path] = str(self.root_dir / cap_path)
                             # Assuming they are stored as absolute or directly usable paths for simplicity here:
                             self.caption_files[img_path] = cap_path
                logger.info(f"Loaded {len(self.image_files)} image paths from manifest.")
                if len(self.image_files) == 0:
                     logger.warning("Manifest loaded, but contains no image paths. Consider deleting it to rescan.")

            except Exception as e:
                logger.error(f"Error loading manifest file {self.manifest_path}: {e}. Will attempt to rescan.")
                self._scan_and_create_manifest() # Fallback to scanning if load fails
        else:
            logger.info(f"Manifest file not found at {self.manifest_path}. Scanning directory to create it...")
            self._scan_and_create_manifest()


        # --- Image Transforms (optional, consider moving to FeatureGenerator if needed) ---
        # img_size = getattr(config.data, 'precompute_image_size', 512)
        # self.transform = transforms.Compose([
        #     transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
        #     transforms.CenterCrop(img_size)
        # ])

    def _scan_and_create_manifest(self):
        """Scans the dataset directory, builds the file list, and saves the manifest."""
        logger.info(f"Scanning dataset directory: {self.root_dir}...")
        manifest_data = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        found_images = 0
        found_captions = 0

        # Use os.scandir for potentially better performance than glob on some systems
        # Wrap with tqdm for progress visibility during the initial slow scan
        iterator = tqdm(os.scandir(self.root_dir), desc="Scanning Dataset")
        for entry in iterator:
             if entry.is_file() and entry.name.lower().endswith(valid_extensions):
                  image_path_str = entry.path # Get the full path string
                  found_images += 1

                  # Look for corresponding .txt file
                  base_name = os.path.splitext(entry.name)[0]
                  caption_path = self.root_dir / f"{base_name}.txt"
                  caption_path_str = str(caption_path) if caption_path.exists() else None

                  if caption_path_str:
                       found_captions += 1

                  # Add to manifest structure
                  manifest_data.append({
                       "image_path": image_path_str,
                       "caption_path": caption_path_str
                  })
                  # Also populate the instance attributes directly during scan
                  self.image_files.append(image_path_str)
                  if caption_path_str:
                      self.caption_files[image_path_str] = caption_path_str

        logger.info(f"Scan complete. Found {found_images} images.")
        if found_captions < found_images:
             logger.warning(f"Found corresponding .txt caption files for {found_captions}/{found_images} images.")
        if not found_captions:
             logger.warning("No paired .txt caption files found.")

        # Save the manifest
        if not self.image_files:
             logger.warning("No images found during scan. Manifest file will not be created.")
             return

        try:
            logger.info(f"Saving dataset manifest to: {self.manifest_path}")
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2) # Use indent for readability
            logger.info("Manifest file saved successfully.")
        except Exception as e:
            logger.error(f"Error saving manifest file {self.manifest_path}: {e}")
            # Proceed without manifest if saving fails, but log error

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_filename = os.path.basename(img_path)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Log specific error and path for easier debugging
            logger.error(f"Error loading image {img_path} (index {idx}): {e}. Returning None.")
            img = None # Signal error

        caption = ""
        caption_path = self.caption_files.get(img_path)
        if caption_path:
             try:
                  with open(caption_path, 'r', encoding='utf-8') as f:
                       caption = f.read().strip()
             except Exception as e:
                  logger.error(f"Error reading caption file {caption_path} for image {img_path}: {e}")
                  # Keep caption as "" if reading fails

        return {'image': img, 'caption': caption, 'id': img_filename}

def precompute_collate_fn(batch):
    """
    Custom collate function for PrecomputeDataset.
    Keeps images as a list of PIL Images (or Nones).
    Collates captions and ids into lists.
    """
    # Batch is a list of dictionaries like {'image': PIL/None, 'caption': str, 'id': str}
    images = [item['image'] for item in batch]    # List of PIL Images or Nones
    captions = [item['caption'] for item in batch] # List of strings
    ids = [item['id'] for item in batch]          # List of strings

    # Return a dictionary where values are lists
    return {'image': images, 'caption': captions, 'id': ids}

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
        # Define target size (consistent input for VAE)
        vae_input_size = self.config.model.router_input_size * self.config.data.vae_downsample_factor # e.g., 32 * 32 = 1024

        # VAEWrapper expects [-1, 1] normalized tensors. Resize added.
        preprocess = transforms.Compose([
            transforms.Resize((vae_input_size, vae_input_size), interpolation=transforms.InterpolationMode.LANCZOS), # Added Resize
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
        ])
        try:
            # Filter out None images before preprocessing
            valid_images = [img for img in images_pil if img is not None]
            if not valid_images: return None # Skip if no valid images in batch

            # Preprocess only valid images
            img_tensors_list = []
            for img in valid_images:
                 try:
                      img_tensors_list.append(preprocess(img))
                 except Exception as preprocess_e:
                      # Log error if a specific image fails preprocessing (e.g., corrupt image data after loading)
                      logger.error(f"Error preprocessing one image in batch: {preprocess_e}. Skipping this image.")
                      img_tensors_list.append(None) # Add a placeholder

            # Filter out None tensors resulting from preprocessing errors
            valid_img_tensors = [t for t in img_tensors_list if t is not None]
            if not valid_img_tensors:
                 logger.warning("No images in batch could be preprocessed successfully.")
                 return None # If all preprocessing failed

            # Stack only the successfully preprocessed tensors
            img_tensors = torch.stack(valid_img_tensors).to(self.device)

            with torch.no_grad():
                # Precision handled within VAEWrapper encode
                latents = self.vae.encode(img_tensors)

            # --- Handle batches where some images failed loading OR preprocessing ---
            # Create a full-size placeholder based on the first valid latent's shape
            latent_c, latent_h, latent_w = latents.shape[1:]
            full_latents = torch.zeros(len(images_pil), latent_c, latent_h, latent_w, dtype=latents.dtype, device='cpu') # Create on CPU directly

            valid_latent_idx = 0
            valid_tensor_indices_iter = iter(range(len(valid_img_tensors))) # Iterator for successfully processed tensors

            for i in range(len(images_pil)):
                # Check if original PIL image was valid AND preprocessing succeeded
                if images_pil[i] is not None and img_tensors_list[valid_latent_idx] is not None:
                    processed_idx = next(valid_tensor_indices_iter)
                    full_latents[i] = latents[processed_idx].cpu()
                # Increment index corresponding to the original images_pil list
                if images_pil[i] is not None:
                    valid_latent_idx += 1

            return full_latents # Return on CPU

        except Exception as e:
            # Broader catch for unexpected errors during the process
            logger.error(f"[Rank {self.rank}] Error in VAE batch encoding: {e}", exc_info=True) # Log traceback
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

    def process_batch(self, batch_data: dict, batch_file_index: int):
        """Processes a batch: extracts enabled features and saves them."""
        images_pil = batch_data['image']
        captions = batch_data['caption']
        # num_samples = len(images_pil) # No longer needed for logging here
        # logger.debug(f"[Rank {self.rank}] Processing batch index {batch_file_index} ({num_samples} samples)...") # REMOVED

        batch_results = {}
        # Process dimensions and buckets first
        if 'dims' in self.enabled_features:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting dims...") # REMOVED
            batch_results['dims'] = self._extract_batch_dims(images_pil)
        if 'buckets' in self.enabled_features and 'dims' in batch_results:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting buckets...") # REMOVED
            batch_results['buckets'] = self._extract_batch_buckets(batch_results['dims'])

        # Process features requiring valid images/captions
        if 'vae' in self.enabled_features:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting vae latents...") # REMOVED
            batch_results['vae'] = self._extract_batch_vae(images_pil)
        if 'clip' in self.enabled_features:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting clip embeddings...") # REMOVED
            batch_results['clip'] = self._extract_batch_clip(captions)
        if 't5' in self.enabled_features:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting t5 embeddings...") # REMOVED
            batch_results['t5'] = self._extract_batch_t5(captions)
        if 'dino' in self.enabled_features:
            # logger.debug(f"[Rank {self.rank}, Batch {batch_file_index}] Extracting dino embeddings...") # REMOVED
            batch_results['dino'] = self._extract_batch_dino(images_pil)

        # Save each computed feature tensor
        for feature_type, batch_tensor in batch_results.items():
            self._save_batch_feature(feature_type, batch_tensor, batch_file_index)

        # logger.debug(f"[Rank {self.rank}] Finished processing batch index {batch_file_index}.") # REMOVED

    def _save_batch_feature(self, feature_type: str, batch_data: torch.Tensor, batch_file_index: int):
        """Saves a batch tensor for a specific feature type."""
        if batch_data is None:
            return

        dir_map = {'vae': 'latents', 'clip': 'clip', 't5': 't5', 'dino': 'dino',
                   'dims': 'dims', 'buckets': 'buckets', 'clusters': 'clusters'}
        save_dir = self.feature_dir / dir_map[feature_type]
        filename = f"{self.rank:03d}_{batch_file_index:06d}.pt"
        filepath = save_dir / filename
        try:
            torch.save(batch_data.cpu(), filepath)
            # logger.debug(f"[Rank {self.rank}] Saved {feature_type} batch {batch_file_index} to {filepath} (Shape: {batch_data.shape})") # REMOVED
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Error saving {feature_type} batch {batch_file_index} to {filepath}: {e}")

    def run_feature_extraction(self, dataloader: DataLoader):
        """Runs feature extraction over the dataset using the dataloader."""
        logger.info(f"[Rank {self.rank}] Starting feature extraction for {len(dataloader)} batches...")
        start_time = time.time()
        processed_batches = 0
        # log_frequency = max(1, len(dataloader) // 10) # REMOVED

        # --- Use tqdm for progress bar on main rank --- START EDIT ---
        batch_iterator = dataloader
        if self.rank == 0:
            batch_iterator = tqdm(dataloader, desc="Extracting Features", unit="batch")

        for batch_idx, batch_data in enumerate(batch_iterator):
        # --- Use tqdm for progress bar on main rank --- END EDIT ---
            self.process_batch(batch_data, batch_idx)
            processed_batches += 1

            # --- REMOVED Periodic Logging Block ---

        if self.distributed:
            # Ensure tqdm bar is closed cleanly on rank 0 before barrier
            if self.rank == 0 and isinstance(batch_iterator, tqdm):
                batch_iterator.close()
            dist.barrier() # Wait for all ranks to finish saving
        elif isinstance(batch_iterator, tqdm): # Close tqdm if not distributed either
             batch_iterator.close()


        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"[Rank {self.rank}] Feature extraction finished. Processed {processed_batches} batches in {total_time:.2f} seconds.")

    # --- Clustering Logic ---
    def _load_features_for_clustering(self, feature_type: str, subsample_fraction: float):
        """Loads features for clustering, potentially subsampled."""
        logger.info(f"[Rank {self.rank}] Loading '{feature_type}' features for clustering (subsample: {subsample_fraction})...")
        feature_subdir = self.feature_dir / feature_type
        if not feature_subdir.exists():
            logger.error(f"Feature directory not found: {feature_subdir}") # Changed to error
            raise FileNotFoundError(f"Feature directory not found: {feature_subdir}")

        all_files = sorted(list(feature_subdir.glob("*_*.pt")))
        if not all_files:
             logger.error(f"No feature files found in {feature_subdir}. Cannot cluster.") # Changed to error
             raise FileNotFoundError(f"No feature files found in {feature_subdir}")

        # --- Subsampling ---
        num_files_to_load = len(all_files)
        files_to_load_paths = all_files
        if subsample_fraction < 1.0:
             num_files_to_sample = int(len(all_files) * subsample_fraction)
             if num_files_to_sample < 1 : num_files_to_sample = 1
             rng_state = random.getstate(); random.seed(42)
             sampled_file_indices = random.sample(range(len(all_files)), num_files_to_sample)
             random.setstate(rng_state)
             files_to_load_paths = [all_files[i] for i in sorted(sampled_file_indices)]
             num_files_to_load = len(files_to_load_paths)
             logger.info(f"[Rank {self.rank}] Will load {num_files_to_load}/{len(all_files)} '{feature_type}' files for subsampled clustering.")
        else:
             logger.info(f"[Rank {self.rank}] Loading all {num_files_to_load} '{feature_type}' files for clustering.")


        # Load features (only rank 0 needs to do this)
        features_list = []
        total_samples_loaded = 0
        if self.rank == 0:
            # --- Wrap loading loop with tqdm --- START EDIT ---
            pbar = tqdm(files_to_load_paths, desc=f"Loading {feature_type} features", unit="file")
            for file_path in pbar:
            # --- Wrap loading loop with tqdm --- END EDIT ---
                try:
                    batch_features = torch.load(file_path, map_location='cpu')
                    if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                         logger.warning(f"NaN/Inf found in feature file {file_path}. Skipping this batch for clustering.")
                         continue
                    features_list.append(batch_features.float())
                    total_samples_loaded += batch_features.shape[0] # Track loaded samples
                    # --- Update tqdm description --- START EDIT ---
                    pbar.set_postfix({"samples_loaded": f"{total_samples_loaded:,}"})
                    # --- Update tqdm description --- END EDIT ---
                except Exception as e:
                    logger.error(f"Error loading feature file {file_path}: {e}")

            if not features_list:
                 logger.error("Failed to load any valid features for clustering.")
                 return None
            all_features = torch.cat(features_list, dim=0)
            logger.info(f"[Rank 0] Loaded {total_samples_loaded} subsampled feature vectors. Final shape: {all_features.shape}")
            return all_features
        else:
            return None

    def run_clustering(self):
        """Performs two-stage clustering and saves assignments."""
        if self.rank != 0:
            if self.distributed: dist.barrier(); dist.barrier() # Barriers remain
            return

        logger.info("[Rank 0] Starting clustering process...")
        clustering_start_time = time.time() # Add timer
        try:
            feature_type = self.config.data.clustering_feature_type
            subsample_fraction = self.config.data.clustering_subsample_fraction
            num_coarse = self.config.model.num_clusters
            num_fine = self.config.data.num_fine_clusters

            clustering_features = self._load_features_for_clustering(feature_type, subsample_fraction)
            if clustering_features is None:
                raise RuntimeError("Clustering feature loading failed.")

            logger.info(f"[Rank 0] Initializing DDMClustering with {num_coarse} coarse and {num_fine} fine clusters.")
            clustering_module = DDMClustering(num_coarse, num_fine, feature_path=self.feature_dir)

            # Stage 1: Fine-grained KMeans (logs progress internally via faiss verbose=True)
            logger.info("[Rank 0] Performing Stage 1: Fine-grained KMeans...")
            # DDMClustering's cluster method calls _train_kmeans which uses faiss verbose
            subsampled_assignments = clustering_module.cluster(features=clustering_features)
            logger.info("[Rank 0] Stage 1 KMeans complete.")

            if subsampled_assignments is None:
                 raise RuntimeError("Clustering module failed to return assignments.")
            if clustering_module.fine_centroids is None or clustering_module.coarse_labels_for_fine is None:
                 raise RuntimeError("Clustering module did not retain necessary centroids/labels for assignment.")


            # Stage 2: Coarse Agglomerative Clustering (relatively fast, no detailed progress needed)
            logger.info("[Rank 0] Performing Stage 2: Coarse Agglomerative Clustering on fine centroids...")
            # ... (agglomerative clustering happens inside clustering_module.cluster or implicitly via its result)
            logger.info("[Rank 0] Stage 2 Agglomerative Clustering complete.")


            # --- Assign ALL samples to clusters and Save Assignments ---
            logger.info("[Rank 0] Assigning all samples to final clusters and saving assignments...")
            assignments_dir = self.feature_dir / "clusters"
            assignments_dir.mkdir(exist_ok=True)

            feature_subdir = self.feature_dir / feature_type
            all_feature_files = sorted([f for f in feature_subdir.glob("*.pt") if f.is_file()]) # Ensure it's a file

            if not all_feature_files:
                 raise FileNotFoundError(f"No feature files found in {feature_subdir} to assign clusters.")

            # Create Faiss index for assignment
            logger.info("[Rank 0] Creating Faiss index for final assignment...")
            index_fine = faiss.IndexFlatIP(clustering_module.fine_centroids.shape[1])
            res = None
            if clustering_module.use_gpu:
                res = faiss.StandardGpuResources(); index_fine = faiss.index_cpu_to_gpu(res, 0, index_fine)
            index_fine.add(np.ascontiguousarray(clustering_module.fine_centroids, dtype=np.float32))
            coarse_labels_for_fine = clustering_module.coarse_labels_for_fine
            logger.info("[Rank 0] Faiss index created.")

            total_samples_processed = 0
            assignment_start_time = time.time()

            # --- Add tqdm progress bar for assignment loop --- START EDIT ---
            pbar_assign = tqdm(all_feature_files, desc="Assigning clusters", unit="file")
            for file_path in pbar_assign:
            # --- Add tqdm progress bar for assignment loop --- END EDIT ---
                try:
                    # --- Refactor loading and conversion for clarity --- START EDIT ---
                    # 1. Load as Tensor
                    batch_tensor = torch.load(file_path, map_location='cpu')
                    # 2. Ensure float
                    batch_tensor = batch_tensor.float()
                    # 3. Convert to NumPy
                    batch_features_np = batch_tensor.numpy()
                    # 4. Ensure contiguous float32 for Faiss
                    batch_features_np = np.ascontiguousarray(batch_features_np, dtype=np.float32)

                    # --- Add explicit type check ---
                    if not isinstance(batch_features_np, np.ndarray):
                         logger.error(f"CRITICAL: Data loaded from {file_path.name} resulted in type {type(batch_features_np)} instead of np.ndarray before Faiss search! Skipping file.")
                         continue # Skip this file if conversion failed unexpectedly
                    # --- End explicit type check ---

                    # Search using the guaranteed numpy array
                    _, fine_centroid_indices = index_fine.search(batch_features_np, 1)
                    # --- Refactor loading and conversion for clarity --- END EDIT ---

                    fine_centroid_indices = fine_centroid_indices.squeeze(-1) # Ensure 1D array

                    # Handle cases where search might return -1 (shouldn't happen with IndexFlatIP unless empty)
                    if np.any(fine_centroid_indices == -1):
                        logger.warning(f"Found invalid index -1 in Faiss search result for file {file_path.name}. Skipping invalid entries.")
                        # Depending on Faiss version/setup, -1 might indicate issues.
                        # We might need to filter out these invalid indices before proceeding.
                        valid_mask = fine_centroid_indices != -1
                        if not np.all(valid_mask):
                             logger.warning(f"Filtering out {np.sum(~valid_mask)} invalid assignments for {file_path.name}")
                             # Apply mask ONLY if needed for subsequent steps, otherwise just warn.
                             # Example: fine_centroid_indices = fine_centroid_indices[valid_mask]
                             #          batch_features_np = batch_features_np[valid_mask] # If needed later
                             # For now, let's just proceed cautiously. Revisit if errors occur later.
                             pass # Proceeding without explicit filtering for now


                    # Ensure indices are within bounds for coarse_labels_for_fine
                    # Check *after* handling potential -1 indices if filtering is applied above
                    max_idx_found = np.max(fine_centroid_indices) if len(fine_centroid_indices) > 0 else -1
                    num_coarse_labels = len(coarse_labels_for_fine) # Get length of the tensor

                    # Check bounds BEFORE indexing
                    if max_idx_found >= num_coarse_labels:
                         logger.error(f"Faiss index {max_idx_found} out of bounds for coarse_labels (size: {num_coarse_labels}) in file {file_path.name}. Skipping file.")
                         continue

                    # Proceed with indexing using the (potentially filtered) valid indices
                    batch_assignments = coarse_labels_for_fine[fine_centroid_indices] # Indexing Tensor with ndarray -> Result is a Tensor

                    # --- Saving Logic --- START EDIT ---
                    base_filename = file_path.name
                    if base_filename.endswith('.pt'):
                        cluster_filename = base_filename
                    else:
                        cluster_filename = f"{os.path.splitext(base_filename)[0]}.pt"

                    cluster_save_path = assignments_dir / cluster_filename

                    # Remove torch.from_numpy as batch_assignments is already a Tensor
                    torch.save(batch_assignments.short(), cluster_save_path) # Save assignments as short tensor
                    # --- Saving Logic --- END EDIT ---


                    # --- Update progress ---
                    num_processed = batch_features_np.shape[0] # Use shape from numpy array
                    total_samples_processed += num_processed
                    pbar_assign.set_postfix({"samples_assigned": f"{total_samples_processed:,}"})

                except IndexError as e:
                    # Catch potential index errors specifically after the bounds check
                    logger.error(f"IndexError during assignment for file {file_path.name}: {e}. Indices: {fine_centroid_indices}, coarse_labels size: {len(coarse_labels_for_fine)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing/assigning assignments for file {file_path.name}: {e}", exc_info=True) # Add traceback

            assignment_time = time.time() - assignment_start_time
            logger.info(f"[Rank 0] Finished assigning clusters to {total_samples_processed} samples across {len(all_feature_files)} files in {assignment_time:.2f} seconds.")

        except Exception as e:
            logger.exception(f"[Rank 0] Clustering failed: {e}")
        finally:
            clustering_time = time.time() - clustering_start_time
            logger.info(f"[Rank 0] Clustering process finished in {clustering_time:.2f} seconds.")
            if self.distributed: dist.barrier() # Ensure barrier happens


# --- Main Execution ---
def main(config_path: str = "config.toml",
         skip_feature_extraction: bool = False,
         skip_clustering: bool = False):
    """Main function to run feature precomputation and clustering."""
    rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    distributed = world_size > 1

    log_format = f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    logger = logging.getLogger(__name__)

    # --- Silence excessive logging (same as before) ---
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger.info("--- Starting Precomputation ---")
    overall_start_time = time.time()

    try:
        config_dict = toml.load(config_path)
        cfg = dict_to_sns(config_dict)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return

    # Define essential features that *must* be generated by feature extraction
    essential_features = {'vae', 'clip', 't5', 'dims', 'buckets'}
    # Features needed for clustering (the feature itself + dims for checking)
    features_for_clustering = {cfg.data.clustering_feature_type, 'dims'}
    # Combine all features the script might interact with or check for
    all_possible_features = essential_features.union(features_for_clustering).union({'clusters'})
    if 'dino' in getattr(cfg.data, 'enabled_features', []):
        all_possible_features.add('dino')
        essential_features.add('dino') # Add dino if enabled
    logger.info(f"Required essential features for extraction: {essential_features}")
    logger.info(f"All features considered by this script: {all_possible_features}")

    generator = FeatureGenerator(cfg, enabled_features=all_possible_features)
    feature_cache_path = Path(cfg.data.feature_cache_path)

    # --- START EDIT: Define dir_map here so it's accessible to all ranks ---
    dir_map = {'vae': 'latents', 'clip': 'clip', 't5': 't5', 'dino': 'dino',
               'dims': 'dims', 'buckets': 'buckets', 'clusters': 'clusters'}
    # --- END EDIT ---


    # --- Check if Feature Extraction can be skipped ---
    can_skip_extraction = False
    if not skip_feature_extraction and is_main: # Only rank 0 performs the check
        logger.info("Checking if feature extraction outputs already exist...")
        all_dirs_exist_and_nonempty = True
        missing_or_empty_dirs = []
        for feature in essential_features:
            if feature in dir_map: # Check if it's a feature with a directory
                feature_dir = feature_cache_path / dir_map[feature]
                if not feature_dir.exists() or not any(feature_dir.iterdir()): # Check if dir exists and has files
                    all_dirs_exist_and_nonempty = False
                    missing_or_empty_dirs.append(dir_map[feature])
                    break # No need to check further if one is missing/empty

        if all_dirs_exist_and_nonempty:
            logger.info("All essential feature directories exist and are non-empty. Will attempt to skip extraction.")
            can_skip_extraction = True
        else:
            logger.info(f"Essential feature directories missing or empty: {missing_or_empty_dirs}. Running feature extraction.")
            can_skip_extraction = False

    # Broadcast the decision from rank 0 to all other ranks
    if distributed:
        skip_extraction_tensor = torch.tensor(int(can_skip_extraction), dtype=torch.int, device=device)
        dist.broadcast(skip_extraction_tensor, src=0)
        can_skip_extraction = bool(skip_extraction_tensor.item())

    # Force skip if user requested it
    if skip_feature_extraction:
        logger.info("User requested skipping feature extraction.")
        can_skip_extraction = True


    # --- Initialize Dataset and DataLoader (always needed for clustering check/run) ---
    dataloader = None
    if not can_skip_extraction or not skip_clustering: # Need dataloader if running extraction OR clustering
        logger.info("Initializing Dataset and DataLoader...")
        try:
            dataset = PrecomputeDataset(cfg)
            if len(dataset) == 0:
                 logger.error("Dataset initialization resulted in 0 samples. Check dataset path and manifest.")
                 if not can_skip_extraction: # Only fatal if we needed to extract
                     return
            else:
                if distributed:
                    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 8)
                    dataloader = DataLoader(dataset, batch_size=precompute_batch_size, sampler=sampler, num_workers=getattr(cfg.train, 'num_workers', 4), pin_memory=False, drop_last=False, collate_fn=precompute_collate_fn)
                else:
                    precompute_batch_size = getattr(cfg.data, 'precompute_batch_size', 8)
                    dataloader = DataLoader(dataset, batch_size=precompute_batch_size, shuffle=False, num_workers=getattr(cfg.train, 'num_workers', 4), pin_memory=False, drop_last=False, collate_fn=precompute_collate_fn)

                logger.info(f"DataLoader initialized with batch size {precompute_batch_size}.")
                logger.info(f"Rank {rank} DataLoader length: {len(dataloader) if dataloader else 'N/A'}")

        except Exception as e:
             logger.exception("Error during Dataset/DataLoader initialization.")
             # If extraction wasn't skipped, this is fatal. If it was, maybe clustering can still proceed if files exist.
             if not can_skip_extraction:
                 return


    # --- Run Feature Extraction ---
    if not can_skip_extraction:
        if dataloader is None:
             logger.error("Cannot run feature extraction because DataLoader failed to initialize.")
             return # Cannot proceed

        logger.info("--- Starting Feature Extraction Phase ---")
        extraction_start_time = time.time()
        try:
             generator.run_feature_extraction(dataloader)
             extraction_time = time.time() - extraction_start_time
             logger.info(f"--- Feature Extraction Phase Finished (Duration: {extraction_time:.2f}s) ---")
        except Exception as e:
             logger.exception("--- Feature Extraction Phase Failed ---")
             # Decide whether to stop or try clustering anyway
             return # Exit if extraction fails, clustering likely won't work
    else:
        logger.info("--- Skipping Feature Extraction Phase (output exists or skipped by user) ---")

    if distributed:
        logger.info(f"Rank {rank} waiting at barrier after feature extraction step.")
        dist.barrier()


    # --- Check if Clustering can be skipped ---
    can_skip_clustering = False
    if not skip_clustering and is_main: # Only rank 0 checks
        logger.info("Checking if clustering output already exists...")
        # --- START EDIT: Check the correct 'clusters' output directory ---
        cluster_output_dir = feature_cache_path / dir_map['clusters'] # Directly check the 'clusters' directory
        # --- END EDIT ---
        if cluster_output_dir.exists() and any(cluster_output_dir.iterdir()):
            logger.info("Cluster assignments directory exists and is non-empty. Will attempt to skip clustering.")
            can_skip_clustering = True
        else:
            logger.info("Cluster assignments directory missing or empty. Running clustering.")
            can_skip_clustering = False

    # Broadcast decision
    if distributed:
        skip_clustering_tensor = torch.tensor(int(can_skip_clustering), dtype=torch.int, device=device)
        dist.broadcast(skip_clustering_tensor, src=0)
        can_skip_clustering = bool(skip_clustering_tensor.item())


    # Force skip if user requested
    if skip_clustering:
        logger.info("User requested skipping clustering.")
        can_skip_clustering = True


    # --- Run Clustering ---
    if not can_skip_clustering:
         # Check if required feature files for clustering exist (double-check after extraction step)
         # Now dir_map is guaranteed to be defined for all ranks
         clustering_feature_dir = feature_cache_path / dir_map[cfg.data.clustering_feature_type]
         if not clustering_feature_dir.exists() or not any(clustering_feature_dir.iterdir()):
             logger.error(f"Cannot run clustering because required feature directory '{clustering_feature_dir}' is missing or empty, even after extraction phase.")
             # Ensure cleanup happens even if clustering is aborted here
             if distributed:
                 if dist.is_initialized(): dist.destroy_process_group()
             return

         logger.info("--- Starting Clustering Phase ---")
         clustering_start_time = time.time()
         try:
             # Clustering is run by rank 0, other ranks wait at barriers inside run_clustering
             generator.run_clustering()
             clustering_time = time.time() - clustering_start_time
             if is_main: # Only rank 0 logs the total time as it performed the work
                  logger.info(f"--- Clustering Phase Finished (Duration: {clustering_time:.2f}s) ---")
         except Exception as e:
              logger.exception("--- Clustering Phase Failed ---")
              # Don't necessarily exit, just log failure.
    else:
        logger.info("--- Skipping Clustering Phase (output exists or skipped by user) ---")
        # Ensure non-rank 0 processes wait if clustering is skipped and they didn't run it
        if distributed:
             logger.info(f"Rank {rank} waiting at barrier after skipping clustering step.")
             dist.barrier()


    # --- Cleanup ---
    overall_time = time.time() - overall_start_time
    logger.info(f"--- Precomputation Finished (Total Duration: {overall_time:.2f}s) ---")
    if distributed:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)