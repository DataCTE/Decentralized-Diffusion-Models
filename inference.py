import torch
import fire
import toml
import os
import numpy as np
from PIL import Image
from types import SimpleNamespace
from typing import List, Optional
import glob

# Project imports (adjust paths if necessary)
from models.router import RouterModel
from models.expert import ExpertModel
from sampling.ddm_sampler import DDMSampler
from data.vae import VAEWrapper # Assuming VAEWrapper for decoding latents
# Import necessary conditioning modules (e.g., CLIP) - adapt based on actual implementation
from data.clip import CLIPTextEncoder # Example, adjust if using flux's HFEmbedder directly
# Helper to load config
from utils import dict_to_sns, load_model_checkpoint, tensor_to_pil, find_latest_checkpoint

# Helper functions (potentially move to a utils file later)
def load_model_checkpoint(model: torch.nn.Module, filepath: str, device: torch.device):
    """Loads state dict from a checkpoint, handling potential 'module.' prefix."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=device)
        # Determine the actual state dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
            if not isinstance(state_dict, dict): state_dict = checkpoint # Assume checkpoint is the state_dict
        elif isinstance(checkpoint, torch.nn.Module):
            state_dict = checkpoint.state_dict()
        else:
             raise TypeError(f"Unsupported checkpoint format at {filepath}.")

        # Handle 'module.' prefix
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            name = k[len("module."):] if k.startswith("module.") else k
            adjusted_state_dict[name] = v

        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)
        if missing_keys: print(f"Warning: Missing keys when loading {model.__class__.__name__}: {missing_keys}")
        if unexpected_keys: print(f"Warning: Unexpected keys when loading {model.__class__.__name__}: {unexpected_keys}")
        print(f"Successfully loaded weights for {model.__class__.__name__} from {filepath}")
    except Exception as e:
        print(f"Error loading checkpoint for {model.__class__.__name__} from {filepath}: {e}")
        raise e

def tensor_to_pil(tensor):
    """Converts a B C H W tensor in range [-1, 1] to a list of PIL Images."""
    # Ensure tensor is on CPU and denormalized
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0 # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    # Convert to HWC uint8 format
    images_np = (tensor.permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]
    return pil_images

def run_inference(
    config_path: str = "config.toml",
    prompt: str = "a photo of an astronaut riding a horse on the moon",
    output_path: str = "output/generated_image.png",
    num_steps: int = 50,
    strategy: str = "top-1", # 'top-1' or 'full'
    seed: int = 1234,
    batch_size: int = 1, # Generate one image by default
    image_height: int = 1024, # Default image size might need adjustment for DC-AE factors
    image_width: int = 1024,
    # Add options for specific checkpoint paths if needed, otherwise use default derived paths
    router_checkpoint: Optional[str] = None,
    expert_checkpoint_pattern: Optional[str] = None, # e.g., "output/checkpoints/expert_{}/expert_*_final.pt"
    find_latest_experts: bool = False # Flag to auto-find latest expert checkpoints
):
    """
    Runs inference using the trained DDM ensemble.

    Args:
        config_path: Path to the TOML configuration file.
        prompt: Text prompt for generation.
        output_path: Path to save the generated image.
        num_steps: Number of diffusion steps for sampling.
        strategy: Sampling strategy ('top-1' or 'full').
        seed: Random seed for noise generation.
        batch_size: Number of images to generate.
        image_height: Desired output image height.
        image_width: Desired output image width.
        router_checkpoint: Optional path to a specific router checkpoint.
        expert_checkpoint_pattern: Optional pattern for expert checkpoints (use {} for expert_id).
        find_latest_experts: If True, automatically finds the latest checkpoint in each expert's dir.
                             Overrides expert_checkpoint_pattern if both are provided.
    """
    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    try:
        config_dict = toml.load(config_path)
        cfg = dict_to_sns(config_dict)
        # Ensure checkpoint_dir is derived (needed for default paths)
        cfg.train.checkpoint_dir = os.path.join(cfg.train.output_dir, "checkpoints")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # --- 2. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Using device: {device}")

    # --- 3. Load Models ---
    # Load Router
    router_cond_dim = getattr(cfg.model, 'router_cond_dim', None)
    router = RouterModel(
        num_clusters=cfg.model.num_clusters,
        input_size=cfg.model.router_input_size,
        patch_size=cfg.model.router_patch_size,
        in_channels=cfg.model.router_in_channels,
        hidden_size=cfg.model.router_hidden_size,
        depth=cfg.model.router_depth,
        num_heads=cfg.model.router_num_heads,
        mlp_ratio=cfg.model.router_mlp_ratio,
        cond_dim=router_cond_dim,
    ).to(device)
    router_ckpt_path = router_checkpoint or find_latest_checkpoint(os.path.join(cfg.train.checkpoint_dir, "router"))
    if not router_ckpt_path or not os.path.exists(router_ckpt_path):
         print(f"Error: Router checkpoint not found at expected path: {router_ckpt_path}")
         return
    load_model_checkpoint(router, router_ckpt_path, device)
    router.eval()

    # Load Experts
    experts: List[ExpertModel] = []
    for i in range(cfg.model.num_clusters):
        print(f"Loading Expert {i}...")
        expert_mmdit_config = {
            'in_channels': cfg.model.expert_in_channels, 'out_channels': cfg.model.expert_out_channels,
            'vec_in_dim': cfg.model.expert_vec_in_dim, 'context_in_dim': cfg.model.expert_context_in_dim,
            'hidden_size': cfg.model.expert_hidden_size, 'mlp_ratio': cfg.model.expert_mlp_ratio,
            'num_heads': cfg.model.expert_num_heads, 'depth': cfg.model.expert_depth,
            'depth_single_blocks': cfg.model.expert_depth_single_blocks, 'axes_dim': cfg.model.expert_axes_dim,
            'theta': cfg.model.expert_theta, 'qkv_bias': cfg.model.expert_qkv_bias,
            'guidance_embed': cfg.model.expert_guidance_embed,
        }
        expert = ExpertModel(mmdit_config=expert_mmdit_config).to(device)

        # Determine expert checkpoint path
        expert_ckpt_path = None
        expert_dir = os.path.join(cfg.train.checkpoint_dir, f"expert_{i}")
        if find_latest_experts:
             expert_ckpt_path = find_latest_checkpoint(expert_dir)
        elif expert_checkpoint_pattern:
             # Try matching the pattern if provided
             pattern_path = expert_checkpoint_pattern.format(i, i) # Assuming two placeholders
             # Use glob to find matching files based on pattern within the expert dir
             matches = glob.glob(os.path.join(expert_dir, os.path.basename(pattern_path)))
             if matches:
                  expert_ckpt_path = max(matches, key=os.path.getmtime) # Get latest match
             else: # Fallback to direct pattern path if glob fails (e.g., absolute path given)
                  if os.path.exists(pattern_path):
                       expert_ckpt_path = pattern_path

        else: # Default: look for 'expert_{i}_final.pt'
             default_path = os.path.join(expert_dir, f"expert_{i}_final.pt")
             if os.path.exists(default_path):
                  expert_ckpt_path = default_path
             else: # If final doesn't exist, try finding latest anyway
                  expert_ckpt_path = find_latest_checkpoint(expert_dir)

        if not expert_ckpt_path or not os.path.exists(expert_ckpt_path):
             print(f"Error: Checkpoint for Expert {i} not found.")
             # Decide whether to error out or continue without this expert
             # For now, let's error out if any expert is missing
             return
             # continue # Or skip this expert

        load_model_checkpoint(expert, expert_ckpt_path, device)
        expert.eval()
        experts.append(expert)

    # Check if the number of loaded experts matches configuration
    if len(experts) != cfg.model.num_clusters:
         print(f"Error: Loaded {len(experts)} experts, but config specifies {cfg.model.num_clusters}. Check checkpoint paths.")
         return

    # --- 4. Initialize Sampler ---
    sampler = DDMSampler(
        router=router,
        experts=experts,
        device=device,
        num_diffusion_timesteps=cfg.train.num_diffusion_timesteps
    )

    # --- 5. Initialize VAE and Conditioning Modules ---
    # VAE for decoding
    # Create VAE config namespace from main cfg
    vae_config_dict = {k: v for k, v in cfg.data.items() if k.startswith('vae_')}
    vae_config_dict['use_mixed_precision'] = cfg.train.use_mixed_precision
    # Add latent_channels to vae_config if not already present under 'vae_' prefix
    if 'latent_channels' not in vae_config_dict:
         vae_config_dict['latent_channels'] = cfg.data.latent_channels
    vae_config = SimpleNamespace(**vae_config_dict)
    vae_wrapper = VAEWrapper(device, vae_config) # VAEWrapper now handles downsample factor internally

    # Conditioning (e.g., CLIP) - Adapt based on your setup
    # Example using the separate CLIPTextEncoder from data/clip.py
    clip_config = SimpleNamespace(
        clip_model_name=getattr(cfg.data, 'clip_model_name', 'openai/clip-vit-large-patch14') # Example default
    )
    clip_encoder = CLIPTextEncoder(device, clip_config)
    # If using Flux HFEmbedder:
    # from models.flux.modules.conditioner import HFEmbedder
    # clip_encoder = HFEmbedder(version="openai/clip-vit-large-patch14", max_length=77).to(device).eval()
    # t5_encoder = HFEmbedder(version="google/flan-t5-xl", max_length=cfg.model.expert_context_in_dim).to(device).eval() # Assuming T5 needed for expert

    # --- 6. Prepare Inputs ---
    # Generate initial noise using the VAE wrapper's method
    latent_height, latent_width = vae_wrapper.get_latent_shape(image_height, image_width)
    # Use latent_channels from the config (now potentially updated by VAEWrapper init)
    noise_shape = (batch_size, cfg.data.latent_channels, latent_height, latent_width)
    xt = torch.randn(noise_shape, device=device)

    # Prepare conditioning dictionary
    conditioning = {}
    prompts = [prompt] * batch_size
    with torch.no_grad():
        # Assuming 'y' is the CLIP pooler output used by both router (if configured) and expert
        clip_embeddings = clip_encoder.encode_pooled(prompts) # Method might vary based on encoder implementation
        conditioning['y'] = clip_embeddings.to(device)

        # If experts require more (like text sequence from T5, img_ids), prepare them here
        # Example for Flux-based experts:
        # t5_embeddings = t5_encoder(prompts) # Get sequence embeddings
        # conditioning['txt'] = t5_embeddings.to(device)
        # # Create txt_ids (dummy or based on actual sequence length)
        # conditioning['txt_ids'] = torch.zeros(batch_size, t5_embeddings.shape[1], 3, device=device)
        # # img_ids would typically be created based on latent shape similar to sampling_flux.py
        # img_h, img_w = latent_height // 2, latent_width // 2 # Check if Flux expects this reduction
        # img_ids = torch.zeros(img_h, img_w, 3)
        # img_ids[..., 1] = img_ids[..., 1] + torch.arange(img_h)[:, None]
        # img_ids[..., 2] = img_ids[..., 2] + torch.arange(img_w)[None, :]
        # conditioning['img_ids'] = torch.zeros(batch_size, img_h * img_w, 3, device=device)

    print(f"Target Image Size: {image_height}x{image_width}")
    print(f"VAE Downsample Factor: {vae_wrapper.downsample_factor}")
    print(f"Generated noise shape (latent): {xt.shape}")
    print(f"Conditioning keys: {list(conditioning.keys())}")

    # --- 7. Run Sampling ---
    print(f"Starting DDM sampling with {num_steps} steps, strategy='{strategy}'...")
    sampled_latents = sampler.sample(
        initial_noise=xt,
        num_steps=num_steps,
        conditioning=conditioning,
        strategy=strategy,
        show_progress=True
    )
    print("Sampling finished.")

    # --- 8. Decode Latents ---
    print("Decoding latents using VAE...")
    with torch.no_grad():
        # Precision handled inside decode method now
        sampled_pixels = vae_wrapper.decode(sampled_latents)
    print("Decoding finished.")

    # --- 9. Save Image(s) ---
    pil_images = tensor_to_pil(sampled_pixels)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if batch_size == 1:
        pil_images[0].save(output_path)
        print(f"Saved generated image to: {output_path}")
    else:
        base, ext = os.path.splitext(output_path)
        for i, img in enumerate(pil_images):
            img_path = f"{base}_{i:03d}{ext}"
            img.save(img_path)
        print(f"Saved {batch_size} generated images to: {base}_*{ext}")


if __name__ == "__main__":
    fire.Fire(run_inference)
