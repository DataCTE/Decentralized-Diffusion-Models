"""Feature extraction for DDM clustering (Paper Section 3.2)"""

import torch
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
            logger.info(f"Starting feature extraction on {total_images:,} images in {total_batches:,} batches")
            logger.info(f"Using device: {self.device}")
            # More accurate estimate based on typical processing speeds (can be adjusted based on hardware)
            est_images_per_sec = 100 if torch.cuda.is_available() else 10  # Rough estimate
            est_minutes = total_images / (est_images_per_sec * 60 * max(1, dist.get_world_size() if dist.is_initialized() else 1))
            logger.info(f"Estimated time: {est_minutes:.1f} minutes ({est_images_per_sec} images/sec)")
            
        # Adaptive logging interval - more frequent updates for small datasets, fewer for large ones
        log_interval = max(1, min(total_batches // 20, 10))  # Log at least 20 times, but not more often than every 10 batches
        
        # Track batch processing times for better estimates
        batch_times = []
        last_update_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_start = time.time()
                
                # Process batch
                try:
                    # Handle different batch formats (dict or tensor)
                    if isinstance(batch, dict):
                        images = batch['image'].to(self.device)
                    else:
                        images = batch.to(self.device)
                    
                    batch_size = len(images)
                    
                    # Extract features
                    batch_features = self.extract_batch(images)
                    features.append(batch_features)
                    processed_images += batch_size
                    
                    # Record batch processing time
                    batch_end = time.time()
                    batch_time = batch_end - batch_start
                    batch_times.append((batch_size, batch_time))
                    
                    # Keep only recent batch times for more accurate estimates
                    if len(batch_times) > 50:
                        batch_times = batch_times[-50:]
                    
                    # Log progress periodically or if it's been a while since last update
                    current_time = time.time()
                    time_since_update = current_time - last_update_time
                    should_update = (batch_idx % log_interval == 0 or 
                                     batch_idx == total_batches - 1 or
                                     time_since_update > 30)  # Force update every 30 seconds
                    
                    if should_log and should_update:
                        # Calculate statistics
                        elapsed = current_time - start_time
                        images_per_sec = processed_images / max(1, elapsed)
                        progress = (batch_idx + 1) / max(1, total_batches) * 100
                        
                        # Calculate ETA based on recent batch times for more accuracy
                        if batch_times:
                            # Calculate average time per image from recent batches
                            total_batch_images = sum(size for size, _ in batch_times)
                            total_batch_time = sum(time for _, time in batch_times)
                            avg_time_per_image = total_batch_time / max(1, total_batch_images)
                            
                            # Estimate remaining time
                            remaining_images = total_images - processed_images
                            eta = avg_time_per_image * remaining_images
                        else:
                            # Fallback to simple estimate
                            eta = (total_batches - batch_idx - 1) / max(1, batch_idx + 1) * elapsed if batch_idx > 0 else 0
                        
                        # Format time as hours:minutes:seconds
                        eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
                        elapsed_str = f"{int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
                        
                        logger.info(f"Feature extraction: {progress:.1f}% complete | "
                                    f"Batch {batch_idx+1}/{total_batches} | "
                                    f"Images: {processed_images:,}/{total_images:,} | "
                                    f"Speed: {images_per_sec:.1f} img/s | "
                                    f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                        
                        last_update_time = current_time
                        
                except Exception as e:
                    logger.error(f"Error in feature extraction at batch {batch_idx}: {str(e)}")
                    continue
        
        # Final stats
        if should_log:
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            logger.info(f"Feature extraction complete: {processed_images:,}/{total_images:,} images processed in {time_str} "
                       f"({processed_images/total_time:.1f} images/sec)")
                
        # Synchronize before continuing
        if dist.is_initialized():
            dist.barrier()
        
        if len(features) == 0:
            logger.warning("No features were extracted! Returning empty tensor.")
            return torch.zeros((0, 1)).numpy()
            
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
