"""VAE wrapper for Decentralized Diffusion Models."""

import torch
from diffusers import AutoencoderDC
import logging
import os
import torch.distributed as dist

logger = logging.getLogger(__name__)

class VAEWrapper:
    """Wrapper for VAE model (now AutoencoderDC) to encode/decode images. LOCKED TO FP32."""
    def __init__(self, device, config):
        self.device = device
        self.config = config # Should be a SimpleNamespace or similar
        self._vae = None  # Lazy loading
        # VAE operations will be forced to float32
        logger.info("VAE operations locked to float32 precision.")

        # Store these crucial parameters from config
        self.vae_model_name_or_path = config.vae_model # Store the name/path
        self.downsample_factor = getattr(config, 'vae_downsample_factor', None)
        if self.downsample_factor is None:
             # Attempt to infer from name if not provided (basic inference)
             if 'f128' in self.vae_model_name_or_path: self.downsample_factor = 128
             elif 'f64' in self.vae_model_name_or_path: self.downsample_factor = 64
             elif 'f32' in self.vae_model_name_or_path: self.downsample_factor = 32
             else: raise ValueError("Could not infer vae_downsample_factor from name and not provided in config.")
             logger.warning(f"Inferred vae_downsample_factor={self.downsample_factor} from model name.")

        # Scaling factor might be loaded from the VAE's config later,
        # but we can keep the config value as a potential override or default.
        self.scaling_factor = getattr(config, 'vae_scaling_factor', None) # Allow None initially

        if getattr(config, 'use_precomputed_latents', False):
            logger.info("VAE loading skipped - using precomputed latents")
            return

        # Trigger lazy loading to check config and potentially load scaling factor
        try:
             self.vae # Access property to load
             if self.scaling_factor is None: # If not overridden in config, use the VAE's config value
                  self.scaling_factor = self._vae.config.scaling_factor
                  logger.info(f"Using scaling_factor from VAE config: {self.scaling_factor}")
             elif abs(self.scaling_factor - self._vae.config.scaling_factor) > 1e-5:
                  logger.warning(f"Config vae_scaling_factor {self.scaling_factor} differs from VAE's internal config scaling_factor {self._vae.config.scaling_factor}. Using value from main config.")
             else:
                  logger.info(f"Using scaling_factor from main config: {self.scaling_factor}")

             # Verify latent channels match
             vae_latent_channels = self._vae.config.latent_channels
             if hasattr(config, 'latent_channels') and config.latent_channels != vae_latent_channels:
                  logger.warning(f"Config latent_channels ({config.latent_channels}) mismatch with VAE latent_channels ({vae_latent_channels}). Ensure model configs match VAE.")
                  # Optionally, update config.latent_channels here if VAE is source of truth
                  # config.latent_channels = vae_latent_channels
             elif not hasattr(config, 'latent_channels'):
                  logger.warning(f"Setting config.latent_channels based on loaded VAE: {vae_latent_channels}")
                  config.latent_channels = vae_latent_channels # Dynamically add if missing

        except Exception as e:
             logger.error(f"Error during VAE initialization/config check: {str(e)}")
             raise

    @property
    def vae(self):
        """Lazy-load VAE only when needed, forced to float32."""
        if self._vae is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] Loading VAE model: {self.vae_model_name_or_path} in float32")

            try:
                 # Load AutoencoderDC using the stored name/path, explicitly setting float32
                 self._vae = AutoencoderDC.from_pretrained(
                      self.vae_model_name_or_path,
                      torch_dtype=torch.float32 # Force float32 loading
                 ).to(self.device) # Move to device

                 self._vae.eval()
                 for param in self._vae.parameters():
                      param.requires_grad = False
                 logger.info(f"AutoencoderDC model loaded successfully to {self.device} in float32.")
                 # Print the loaded config scaling factor for verification
                 logger.info(f"VAE internal config scaling_factor: {self._vae.config.scaling_factor}")
                 logger.info(f"VAE internal config latent_channels: {self._vae.config.latent_channels}")

            except Exception as e:
                 logger.error(f"Failed to load AutoencoderDC model '{self.vae_model_name_or_path}': {e}")
                 raise RuntimeError(f"Failed to load VAE: {e}")

        return self._vae

    @torch.no_grad() # Ensure no gradients computed here
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input image tensor into latents using float32."""
        if self._vae is None: raise RuntimeError("VAE not loaded.")
        
        # Ensure input is float32
        x = x.to(self.device, dtype=torch.float32)
        
        # VAE operates in float32
        latents = self.vae.encode(x).latent_dist.sample()
        
        # Scale latents *after* sampling
        if self.scaling_factor is None:
             raise ValueError("VAE scaling_factor is not set.")
        scaled_latents = latents * self.scaling_factor
        
        return scaled_latents # Already float32

    @torch.no_grad() # Ensure no gradients computed here
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents back to the pixel space using float32."""
        if self._vae is None: raise RuntimeError("VAE not loaded.")
        if self.scaling_factor is None: raise ValueError("VAE scaling_factor is not set.")

        # Ensure latents are float32 on the correct device before scaling/decoding
        latents = latents.to(self.device, dtype=torch.float32)

        # Scale latents back before decoding
        latents = latents / self.scaling_factor

        # VAE operates in float32
        # Handle potential OOM by processing in batches if needed (optional)
        batch_size_limit = getattr(self.config, 'vae_decode_batch_size', 8) # Configurable batch limit
        if latents.shape[0] > batch_size_limit:
            images = []
            for i in range(0, latents.shape[0], batch_size_limit):
                batch_latents = latents[i:i+batch_size_limit]
                batch_images = self.vae.decode(batch_latents).sample
                images.append(batch_images)
            images = torch.cat(images, dim=0)
        else:
            images = self.vae.decode(latents).sample

        # Return decoded images (already float32)
        return images

    def get_latent_shape(self, pixel_height, pixel_width):
        """Calculate latent shape based on the downsample factor."""
        if self.downsample_factor is None:
             raise ValueError("VAE downsample_factor is not set.")
        latent_height = pixel_height // self.downsample_factor
        latent_width = pixel_width // self.downsample_factor
        return (latent_height, latent_width)

    def get_pixel_shape(self, latent_height, latent_width):
        """Calculate pixel dimensions based on the downsample factor."""
        if self.downsample_factor is None:
             raise ValueError("VAE downsample_factor is not set.")
        pixel_height = latent_height * self.downsample_factor
        pixel_width = latent_width * self.downsample_factor
        return (pixel_height, pixel_width)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space"""
        return self.encode(x) 