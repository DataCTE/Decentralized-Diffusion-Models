"""Feature extraction for DDM clustering (Paper Section 3.2)"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import torch.distributed as dist
import time

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Base class for feature extraction in DDM"""
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @torch.no_grad()
    def extract_batch(self, batch):
        """Extract features from a batch of images"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def extract_features(self, dataloader, log_progress=True):
        """Extract features from all images in a dataloader"""
        features = []
        total_batches = len(dataloader)
        start_time = time.time()
        processed_images = 0
        total_images = len(dataloader.dataset)
        
        # Only log from rank 0
        should_log = log_progress and (not dist.is_initialized() or dist.get_rank() == 0)
        
        if should_log:
            logger.info(f"Starting feature extraction on {total_images:,} images")
            logger.info(f"Process will take approximately {total_images * 0.01 / max(1, dist.get_world_size()):.1f} minutes (estimate)")
            
        log_interval = max(1, total_batches // 20)  # Log 20 times during extraction
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Log progress periodically
                if should_log and (batch_idx % log_interval == 0 or batch_idx == total_batches - 1):
                    elapsed = time.time() - start_time
                    images_per_sec = processed_images / max(1, elapsed)
                    progress = batch_idx / max(1, total_batches) * 100
                    eta = (total_batches - batch_idx) / max(1, batch_idx) * elapsed if batch_idx > 0 else 0
                    logger.info(f"Feature extraction: {progress:.1f}% complete | "
                                f"Batch {batch_idx+1}/{total_batches} | "
                                f"Images: {processed_images:,}/{total_images:,} | "
                                f"Speed: {images_per_sec:.1f} img/s | "
                                f"ETA: {eta/60:.1f} min")
                
                # Process batch
                try:
                    # Handle different batch formats (dict or tensor)
                    if isinstance(batch, dict):
                        images = batch['image'].to(self.device)
                    else:
                        images = batch.to(self.device)
                        
                    # Extract features
                    batch_features = self.extract_batch(images)
                    features.append(batch_features)
                    processed_images += len(images)
                except Exception as e:
                    logger.error(f"Error in feature extraction at batch {batch_idx}: {str(e)}")
                    continue
        
        # Final stats
        if should_log:
            total_time = time.time() - start_time
            logger.info(f"Feature extraction complete: {processed_images:,} images processed in {total_time/60:.1f} minutes "
                      f"({processed_images/total_time:.1f} images/sec)")
                
        # Synchronize before continuing
        if dist.is_initialized():
            dist.barrier()
        
        return torch.cat(features).cpu().numpy()


class DINOv2FeatureExtractor(FeatureExtractor):
    """Implements paper's feature extraction using DINOv2"""
    def __init__(self, device=None):
        super().__init__(device)
        self.model = self._init_dinov2()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
    def _init_dinov2(self):
        """Load DINOv2 model with paper-recommended settings"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.no_grad()
    def extract_batch(self, x):
        """Extract features from a batch of images"""
        # Normalize input
        x = (x - self.mean) / self.std
        
        # Extract features
        features = self.model.forward_features(x)['x_norm_patchtokens']
        
        # Average pooling across spatial dimensions
        return features.mean(dim=1)

# For potential future feature extractors
class CLIPFeatureExtractor(FeatureExtractor):
    """Feature extraction using CLIP image encoder"""
    def __init__(self, device=None, model_name="openai/clip-vit-large-patch14"):
        super().__init__(device)
        from transformers import CLIPModel
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def extract_batch(self, x):
        """Extract features using CLIP"""
        # CLIP expects pixel values in range [0, 1]
        x = (x + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        features = self.model.get_image_features(x)
        return features
