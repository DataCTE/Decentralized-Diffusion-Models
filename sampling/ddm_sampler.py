# sampling/sample_ddm.py

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import math
import fire
import toml
from pathlib import Path
from typing import Optional, List, Dict
from types import SimpleNamespace
import random

# Project imports (adjust paths if necessary)
from models.router import RouterModel
from models.expert import ExpertModel, FluxParams
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from data.t5 import T5TextEncoder
from sampling.ddm_sampler import DDMSampler
from utils import (
    dict_to_sns,
    load_model_checkpoint,
    find_latest_checkpoint,
    tensor_to_pil
)
from utils.logging import setup_distributed_logger # For consistent logging style

# Basic logger setup (non-distributed for sampling)
logger = setup_distributed_logger("DDMSampling", rank=0)

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_ids(h: int, w: int, patch_size: int, t5_len: int) -> Dict[str, torch.Tensor]:
    """
    Generates img_ids and txt_ids based on dimensions and patch size.
    Args:
        h (int): Latent height.
        w (int): Latent width.
        patch_size (int): Patch size used by experts.
        t5_len (int): Sequence length of T5 embeddings.
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing 'img_ids' and 'txt_ids'.
    """
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Latent dimensions ({h}x{w}) not divisible by patch size ({patch_size}).")

    grid_h, grid_w = h // patch_size, w // patch_size
    num_img_patches = grid_h * grid_w

    img_ids = torch.zeros(grid_h, grid_w, 3, dtype=torch.float32)
    img_ids[..., 0] = torch.arange(grid_h, dtype=torch.float32)[:, None] # y-coordinates
    img_ids[..., 1] = torch.arange(grid_w, dtype=torch.float32)[None, :] # x-coordinates
    img_ids_flat = img_ids.view(num_img_patches, 3) # Reshape to [N_patches, 3]

    txt_ids = torch.zeros(t5_len, 3, dtype=torch.float32)
    txt_ids[..., 0] = torch.arange(t5_len, dtype=torch.float32) # Sequence position

    return {'img_ids': img_ids_flat, 'txt_ids': txt_ids}

    def sample(
    config_path: str = "config.toml",
    checkpoint_base_dir: str = "output/checkpoints", # Base dir where router/expert_i dirs are
    prompt: str = "A beautiful landscape painting.",
    output_path: str = "output/ddm_sample.png",
    output_height: int = 1024,
    output_width: int = 1024,
    num_steps: int = 50,
    batch_size: int = 1, # Typically 1 for sampling, increase if VRAM allows
    seed: Optional[int] = None,
        strategy: str = 'top-1', # 'top-1' or 'full'
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
    Generates an image using a trained Decentralized Diffusion Model ensemble.

        Args:
        config_path (str): Path to the TOML configuration file used for training.
        checkpoint_base_dir (str): Path to the base directory containing checkpoint subdirs
                                   (e.g., 'output/checkpoints' which contains 'router' and 'expert_0', 'expert_1', ...).
        prompt (str): The text prompt for generation.
        output_path (str): Path to save the generated image.
        output_height (int): Height of the desired output image (pixels). Must be multiple of VAE downsample factor.
        output_width (int): Width of the desired output image (pixels). Must be multiple of VAE downsample factor.
        num_steps (int): Number of DDIM/DDPM inference steps.
        batch_size (int): Number of samples to generate in parallel.
        seed (Optional[int]): Random seed for reproducibility. If None, uses a random seed.
        strategy (str): DDM inference strategy ('top-1' or 'full').
        device (str): Device to run inference on ('cuda' or 'cpu').
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    logger.info(f"Using seed: {seed}")
    set_seed(seed)

    # --- 1. Load Configuration & Setup ---
    logger.info(f"Loading configuration from: {config_path}")
    try:
        config_dict = toml.load(config_path)
        cfg = dict_to_sns(config_dict)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    device = torch.device(device)
    logger.info(f"Using device: {device}")
    base_ckpt_path = Path(checkpoint_base_dir)
    if not base_ckpt_path.exists():
        logger.error(f"Checkpoint base directory not found: {base_ckpt_path}")
        return

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Configuration Consistency Checks ---
    logger.info("Verifying configuration parameters...")
    vae_downsample = cfg.data.vae_downsample_factor
    if output_height % vae_downsample != 0 or output_width % vae_downsample != 0:
        logger.error(f"Output dimensions ({output_height}x{output_width}) must be divisible by VAE downsample factor ({vae_downsample}).")
        return
    latent_height = output_height // vae_downsample
    latent_width = output_width // vae_downsample
    latent_channels = cfg.data.latent_channels
    expert_patch_size = cfg.model.expert_patch_size # Get from model config
    if latent_height % expert_patch_size != 0 or latent_width % expert_patch_size != 0:
         logger.warning(f"Latent dimensions ({latent_height}x{latent_width}) not perfectly divisible by expert patch size ({expert_patch_size}). Ensure this is intended or adjust output size.")
         # Potentially raise error if strict divisibility is required by ExpertModel/patching

    # --- 2. Initialize Models & Load Checkpoints ---
    logger.info("Initializing models and loading checkpoints...")

    # -- Router --
    router_ckpt_dir = base_ckpt_path / "router"
    latest_router_ckpt = find_latest_checkpoint(str(router_ckpt_dir), pattern="router_*.pt") # Specific pattern
    if not latest_router_ckpt:
        logger.error(f"Router checkpoint not found in {router_ckpt_dir}")
        return
    try:
        logger.info("Initializing Router...")
        router = RouterModel(
            num_clusters=cfg.model.num_clusters,
            input_size=cfg.model.router_input_size, # Use config value
            patch_size=cfg.model.router_patch_size,
            in_channels=latent_channels, # Should match VAE output channels
            hidden_size=cfg.model.router_hidden_size,
            depth=cfg.model.router_depth,
            num_heads=cfg.model.router_num_heads,
            mlp_ratio=cfg.model.router_mlp_ratio,
            cond_dim=getattr(cfg.model, 'router_cond_dim', None),
        )
        logger.info(f"Loading router checkpoint: {latest_router_ckpt}")
        load_model_checkpoint(router, latest_router_ckpt, device)
        router = router.to(device).eval()
        logger.info("Router loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize or load router: {e}", exc_info=True)
        return

    # -- Experts --
    experts: List[ExpertModel] = []
    num_experts = cfg.model.num_clusters
    logger.info(f"Loading {num_experts} Expert models...")
    for i in range(num_experts):
        expert_ckpt_dir = base_ckpt_path / f"expert_{i}"
        # Find checkpoint like 'expert_i_final_step_*.pt' or 'expert_i_step_*.pt'
        latest_expert_ckpt = find_latest_checkpoint(str(expert_ckpt_dir), pattern=f"expert_{i}_*.pt")
        if not latest_expert_ckpt:
            logger.error(f"Checkpoint for Expert {i} not found in {expert_ckpt_dir}")
            return
        try:
            # Prepare FluxParams for this expert
            expert_raw_config = {}
            prefix = "expert_"
            for k in dir(cfg.model):
                 if k.startswith(prefix):
                      v = getattr(cfg.model, k)
                      if k != 'expert_patch_size': # Patch size handled separately
                           new_key = k[len(prefix):]
                           expert_raw_config[new_key] = v
            flux_params_for_expert = FluxParams(**expert_raw_config)

            logger.info(f"Initializing Expert {i}...")
            expert = ExpertModel(mmdit_params=flux_params_for_expert)
            logger.info(f"Loading expert {i} checkpoint: {latest_expert_ckpt}")
            load_model_checkpoint(expert, latest_expert_ckpt, device)
            expert = expert.to(device).eval()
            experts.append(expert)
            logger.info(f"Expert {i} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize or load expert {i}: {e}", exc_info=True)
            return

    # -- VAE --
    try:
        logger.info("Initializing VAE...")
        vae_sub_config = SimpleNamespace(
             **{k: v for k, v in cfg.data.__dict__.items() if k.startswith('vae_')},
             use_mixed_precision=cfg.train.use_mixed_precision, # Needed? Maybe not for inference
             latent_channels=latent_channels
        )
        vae = VAEWrapper(device, vae_sub_config) # Assumes VAE doesn't need separate checkpoint loading here
        logger.info("VAE initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize VAE: {e}", exc_info=True)
        return

    # -- Text Encoders --
    try:
        logger.info("Initializing Text Encoders (CLIP, T5)...")
        clip_sub_config = SimpleNamespace(
             clip_model_name=cfg.data.clip_model_name,
             clip_max_token_length=getattr(cfg.data, 'clip_max_token_length', 77),
             use_mixed_precision=cfg.train.use_mixed_precision # Or set false for inference?
        )
        t5_sub_config = SimpleNamespace(
             t5_model_name=cfg.data.t5_model_name,
             t5_max_token_length=getattr(cfg.data, 't5_max_token_length', 128),
             use_mixed_precision=cfg.train.use_mixed_precision # Or set false for inference?
        )
        clip_encoder = CLIPTextEncoder(device, clip_sub_config)
        t5_encoder = T5TextEncoder(device, t5_sub_config)
        logger.info("Text Encoders initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Text Encoders: {e}", exc_info=True)
        return

    # --- 3. Prepare Conditioning ---
    logger.info("Preparing conditioning...")
    with torch.no_grad():
        clip_embed = clip_encoder.encode_pooled([prompt] * batch_size) # [B, cond_dim]
        t5_embed = t5_encoder.encode([prompt] * batch_size)           # [B, seq_len, cond_dim]

    # Generate IDs based on target latent size and expert patch size
    t5_seq_len = t5_embed.shape[1]
    ids = generate_ids(latent_height, latent_width, expert_patch_size, t5_seq_len)
    img_ids_unbatched = ids['img_ids'] # [NumPatches, 3]
    txt_ids_unbatched = ids['txt_ids'] # [SeqLen, 3]

    conditioning = {
        'y': clip_embed.to(device),       # CLIP condition for router and experts
        'txt': t5_embed.to(device),       # T5 condition for experts
        # Expand IDs to batch size
        'img_ids': img_ids_unbatched.unsqueeze(0).expand(batch_size, -1, -1).to(device),
        'txt_ids': txt_ids_unbatched.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    }
    logger.info("Conditioning prepared.")

    # --- 4. Initialize Sampler ---
    logger.info("Initializing DDMSampler...")
    sampler = DDMSampler(
        router=router,
        experts=experts,
        device=device,
        patch_size=expert_patch_size, # Pass patch size from config
        num_diffusion_timesteps=cfg.train.num_diffusion_timesteps,
        beta_start=cfg.train.beta_start,
        beta_end=cfg.train.beta_end,
    )
    logger.info("DDMSampler initialized.")

    # --- 5. Generate Initial Noise ---
    initial_noise = torch.randn(
        batch_size,
        latent_channels,
        latent_height,
        latent_width,
        device=device
    )
    logger.info(f"Generated initial noise: {initial_noise.shape}")

    # --- 6. Run Sampling ---
    logger.info(f"Starting DDM sampling ({num_steps} steps, strategy: {strategy})...")
    try:
        with torch.inference_mode(): # Ensure no gradients are computed
            # AMP context might be beneficial if models support float16
            with torch.autocast(device_type=device.type, enabled=cfg.train.use_mixed_precision):
                sampled_latents = sampler.sample(
                    initial_noise=initial_noise,
                    num_steps=num_steps,
                    conditioning=conditioning,
                    strategy=strategy,
                    show_progress=True
                )
        logger.info("Sampling complete.")
    except Exception as e:
        logger.error(f"Error during sampling: {e}", exc_info=True)
        return

    # --- 7. Decode Latents ---
    logger.info("Decoding latents...")
    try:
        # VAE decoding often done in float32 for precision
        sampled_latents = sampled_latents.float()
        with torch.no_grad():
            decoded_images = vae.decode(sampled_latents) # VAEWrapper handles precision internally
        logger.info("Decoding complete.")
    except Exception as e:
        logger.error(f"Error during VAE decoding: {e}", exc_info=True)
        return

    # --- 8. Save Output ---
    logger.info(f"Saving image to: {output_path}")
    try:
        # Convert tensor to PIL image(s)
        pil_images = tensor_to_pil(decoded_images) # Assumes tensor_to_pil handles B C H W -> list[PIL]
        # Save the first image in the batch (or loop if batch_size > 1)
        if pil_images:
            pil_images[0].save(output_path)
            logger.info("Image saved successfully.")
        else:
            logger.error("tensor_to_pil returned no images.")
    except Exception as e:
        logger.error(f"Error saving image: {e}", exc_info=True)

if __name__ == "__main__":
    fire.Fire(sample)