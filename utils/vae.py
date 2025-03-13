"""VAE wrapper for Decentralized Diffusion Models."""

import torch
from diffusers import AutoencoderKL

class VAEWrapper:
    """Wrapper for VAE model to encode/decode images for latent diffusion"""
    def __init__(self, device, config):
        self.device = device
        
        # Load VAE from specified repository
        self.vae = AutoencoderKL.from_pretrained(
            config.vae_model,
            torch_dtype=torch.float16  # Use half precision for efficiency
        ).to(device)
        
        # Set to evaluation mode and freeze parameters
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)
            
        # VAE scaling factor (commonly used with Stable Diffusion VAEs)
        self.scaling_factor = 0.18215
        
    def encode(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            # Scale latents according to VAE convention
            latents = latents * self.scaling_factor
            return latents
    
    def decode(self, latents):
        """Decode latents to pixel space"""
        with torch.no_grad():
            # Scale latents back
            latents = latents / self.scaling_factor
            # Decode latents to images
            images = self.vae.decode(latents).sample
            return images 