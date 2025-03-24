"""VAE wrapper for Decentralized Diffusion Models."""

import torch
from diffusers import AutoencoderKL
import logging
import os

logger = logging.getLogger(__name__)

class VAEWrapper:
    """Wrapper for VAE model to encode/decode images for latent diffusion"""
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.precision = torch.float16 if config.use_mixed_precision else torch.float32
        
        # Handle model loading paths
        vae_model_path = config.vae_model
        if not os.path.exists(vae_model_path) and "/" not in vae_model_path:
            # Assume it's a HuggingFace model ID
            logger.info(f"Loading VAE from HuggingFace: {vae_model_path}")
        else:
            logger.info(f"Loading VAE from local path: {vae_model_path}")
        
        try:
            # Load VAE from specified repository
            self.model = AutoencoderKL.from_pretrained(vae_model_path).to(device)
            self.model.eval()
            
            # Explicitly convert model weights to match precision
            if config.use_mixed_precision:
                self.model.half()
            
            # VAE scaling factor (commonly used with Stable Diffusion VAEs)
            self.scaling_factor = getattr(config, 'vae_scaling_factor', 0.18215)
            
            logger.info(f"VAE loaded successfully with scaling factor {self.scaling_factor}")
            
        except Exception as e:
            logger.error(f"Error loading VAE: {str(e)}")
            raise RuntimeError(f"Failed to load VAE: {str(e)}")
            
    def encode(self, x):
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            x = x.to(self.device, dtype=self.precision)
            return self.model.encode(x).latent_dist.sample().to(torch.float32)
    
    def decode(self, latents):
        """
        Decode latents to pixel space
        
        Args:
            latents: [B, C, H/8, W/8] tensor of latents
            
        Returns:
            [B, 3, H, W] tensor of images in range [-1, 1]
        """
        # Ensure latents are on the correct device
        if latents.device != self.device:
            latents = latents.to(self.device)
            
        with torch.no_grad():
            # Scale latents back
            latents = latents / self.scaling_factor
            
            # Handle potential OOM by processing in batches if needed
            if latents.shape[0] > 8 and hasattr(self.config, 'vae_batch_size'):
                # Process in batches to avoid OOM
                batch_size = self.config.vae_batch_size
                images = []
                
                for i in range(0, latents.shape[0], batch_size):
                    batch_latents = latents[i:i+batch_size]
                    batch_images = self.model.decode(batch_latents).sample
                    images.append(batch_images)
                    
                images = torch.cat(images, dim=0)
            else:
                # Process all latents at once
                images = self.model.decode(latents).sample
                
            return images
            
    def get_latent_shape(self, pixel_height, pixel_width):
        """
        Calculate latent shape for a given pixel dimensions
        
        Args:
            pixel_height: Height in pixels
            pixel_width: Width in pixels
            
        Returns:
            (latent_height, latent_width) tuple
        """
        # VAE typically downsamples by a factor of 8
        latent_height = pixel_height // 8
        latent_width = pixel_width // 8
        return (latent_height, latent_width)
        
    def get_pixel_shape(self, latent_height, latent_width):
        """
        Calculate pixel dimensions for given latent shape
        
        Args:
            latent_height: Height in latent space
            latent_width: Width in latent space
            
        Returns:
            (pixel_height, pixel_width) tuple
        """
        # VAE typically upsamples by a factor of 8
        pixel_height = latent_height * 8
        pixel_width = latent_width * 8
        return (pixel_height, pixel_width)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space"""
        return self.encode(x) 