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
        
        # Handle model loading paths
        vae_model_path = config.vae_model
        if not os.path.exists(vae_model_path) and "/" not in vae_model_path:
            # Assume it's a HuggingFace model ID
            logger.info(f"Loading VAE from HuggingFace: {vae_model_path}")
        else:
            logger.info(f"Loading VAE from local path: {vae_model_path}")
        
        try:
            # Load VAE from specified repository
            self.vae = AutoencoderKL.from_pretrained(
                vae_model_path,
                torch_dtype=torch.float16 if hasattr(config, 'use_mixed_precision') and config.use_mixed_precision else torch.float32
            ).to(device)
            
            # Set to evaluation mode and freeze parameters
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad_(False)
                
            # VAE scaling factor (commonly used with Stable Diffusion VAEs)
            self.scaling_factor = getattr(config, 'vae_scaling_factor', 0.18215)
            
            logger.info(f"VAE loaded successfully with scaling factor {self.scaling_factor}")
            
        except Exception as e:
            logger.error(f"Error loading VAE: {str(e)}")
            raise RuntimeError(f"Failed to load VAE: {str(e)}")
            
    def encode(self, images):
        """
        Encode images to latent space
        
        Args:
            images: [B, C, H, W] tensor of images in range [-1, 1]
            
        Returns:
            [B, C, H/8, W/8] tensor of latents
        """
        # Add graph-safe memory handling
        if images.device != self.device:
            images = images.to(self.device)
            
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=images.dtype == torch.float16):
            # Handle potential OOM by processing in batches if needed
            if images.shape[0] > 8 and hasattr(self.config, 'vae_batch_size'):
                # Process in batches to avoid OOM
                batch_size = self.config.vae_batch_size
                latents = []
                
                for i in range(0, images.shape[0], batch_size):
                    batch_images = images[i:i+batch_size]
                    batch_latents = self.vae.encode(batch_images).latent_dist.sample()
                    latents.append(batch_latents)
                    
                latents = torch.cat(latents, dim=0)
            else:
                # Process all images at once
                latents = self.vae.encode(images).latent_dist.sample()
                
            # Scale latents according to VAE convention
            latents = latents * self.scaling_factor
            return latents
    
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
                    batch_images = self.vae.decode(batch_latents).sample
                    images.append(batch_images)
                    
                images = torch.cat(images, dim=0)
            else:
                # Process all latents at once
                images = self.vae.decode(latents).sample
                
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