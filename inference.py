"""Inference script for Decentralized Diffusion Models."""

import os
import torch
import argparse
import math
from tqdm import tqdm
from PIL import Image
import numpy as np
import wandb
import logging
import threading
from queue import Queue

# Setup logging with non-blocking handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Non-blocking logging queue
log_queue = Queue()
def log_worker():
    while True:
        record = log_queue.get()
        if record is None:
            break
        logger.handle(record)

# Start logging thread
log_thread = threading.Thread(target=log_worker)
log_thread.daemon = True
log_thread.start()

# Helper function for non-blocking wandb logging
def log_to_wandb_async(data):
    """
    Log data to wandb asynchronously to avoid blocking the main thread.
    
    Args:
        data: Dictionary of data to log to wandb
    """
    if wandb.run is not None:
        threading.Thread(
            target=wandb.log,
            args=(data,),
            daemon=True
        ).start()

from config import DDMConfig
from models.dit import ExpertDiT
from models.router import RouterModel
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with DDM")
    parser.add_argument("--prompt", type=str, default="1girl", help="Text prompt for generation")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ddm", help="Directory with model checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save generated images")
    parser.add_argument("--use_distilled", action="store_true", help="Use distilled model instead of ensemble")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--log_to_wandb", action="store_true", help="Log metrics to Weights & Biases")
    return parser.parse_args()

def ddm_sample(
    config,
    router_model,
    expert_models,
    vae_wrapper,
    text_embeddings,
    num_steps=50,
    batch_size=1,
    device="cuda",
    guidance_scale=7.5,
    uncond_embeddings=None,
    log_to_wandb=False,
):
    """
    Sample from the Decentralized Diffusion Model using the flow matching approach.
    This implements the sampling approach described in Section 3.4 of the paper.
    
    The sampling follows Algorithm 2 from the paper:
    1. Initialize x_1 with Gaussian noise
    2. For each timestep t from 1 to 0:
       a. Get router probabilities p_k(x_t, t) for each expert
       b. Select the expert with highest probability (or combine experts)
       c. Compute the flow field u_t(x_t) using the selected expert(s)
       d. Update x_t using the flow field: x_{t-dt} = x_t - u_t(x_t) * dt
    3. Return the final x_0
    
    Args:
        config: Configuration object
        router_model: Router model for expert selection
        expert_models: List of expert models
        vae_wrapper: VAE wrapper for encoding/decoding
        text_embeddings: Text embeddings for conditioning
        num_steps: Number of sampling steps
        batch_size: Batch size
        device: Device to use
        guidance_scale: Scale for classifier-free guidance
        uncond_embeddings: Unconditional embeddings for classifier-free guidance
        log_to_wandb: Whether to log expert usage statistics to wandb
    
    Returns:
        Decoded images
    """
    # Initialize latent variable with random noise
    # The latent space has dimensions [batch_size, latent_channels, height/8, width/8]
    latent = torch.randn(
        batch_size, config.latent_channels, config.image_size // 8, config.image_size // 8
    ).to(device)
    
    # Initialize expert usage counter
    expert_usage_counts = {i: 0 for i in range(len(expert_models))}
    
    # Sampling loop as described in Section 3.4
    for i in tqdm(range(num_steps), desc="Sampling"):
        # Normalized timestep from 1.0 to 0.0 (reverse process)
        t = 1.0 - i / (num_steps - 1)
        t_tensor = torch.ones(batch_size, device=device) * t
        
        # Get router probabilities for expert selection
        with torch.no_grad():
            router_logits = router_model(latent, t_tensor)
            router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        
        # Select expert with highest probability for each sample in batch
        expert_indices = torch.argmax(router_probs, dim=-1)
        
        # Update expert usage counts
        for idx in expert_indices.cpu().numpy():
            expert_usage_counts[idx] += 1
        
        # Compute flow field using selected expert for each sample
        flow = torch.zeros_like(latent)
        for b in range(batch_size):
            expert_idx = expert_indices[b].item()
            # Get the selected expert model
            expert_model = expert_models[expert_idx]
            
            # Prepare inputs for the expert model
            sample_latent = latent[b:b+1]
            sample_t = t_tensor[b:b+1]
            
            # Apply classifier-free guidance if specified
            if guidance_scale > 1.0 and uncond_embeddings is not None:
                # Concatenate unconditional and conditional embeddings
                text_emb = torch.cat([uncond_embeddings, text_embeddings])
                # Duplicate the latent and timestep
                sample_latent_repeat = sample_latent.repeat(2, 1, 1, 1)
                sample_t_repeat = sample_t.repeat(2)
                
                # Get predictions from the expert model
                pred_flow = expert_model(sample_latent_repeat, sample_t_repeat, text_emb)
                
                # Split predictions for unconditional and conditional paths
                uncond_flow, cond_flow = pred_flow.chunk(2)
                
                # Apply classifier-free guidance formula: pred = uncond + scale * (cond - uncond)
                pred_flow = uncond_flow + guidance_scale * (cond_flow - uncond_flow)
                
                # Update the flow for this sample
                flow[b:b+1] = pred_flow
            else:
                # Standard prediction without guidance
                flow[b:b+1] = expert_model(sample_latent, sample_t, text_embeddings)
        
        # Update latent using the flow field (Euler step)
        # This follows the flow matching ODE: dx/dt = f(x,t)
        # For a small step dt, we have: x_{t-dt} = x_t - f(x_t,t) * dt
        dt = 1.0 / (num_steps - 1)
        latent = latent - flow * dt
    
    # Log expert usage statistics
    total_samples = batch_size * num_steps
    logger.info("Expert usage statistics:")
    for expert_idx, count in expert_usage_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Expert {expert_idx}: {count} steps ({percentage:.2f}%)")
    
    # Log to wandb if requested (non-blocking)
    if log_to_wandb and wandb.run is not None:
        expert_usage_percentages = {f"sampling/expert_{idx}_usage": (count / total_samples) * 100 
                                   for idx, count in expert_usage_counts.items()}
        expert_usage_percentages["sampling/num_steps"] = num_steps
        expert_usage_percentages["sampling/batch_size"] = batch_size
        
        # Use helper function for non-blocking logging
        log_to_wandb_async(expert_usage_percentages)
    
    # Decode the final latent to get images
    with torch.no_grad():
        images = vae_wrapper.decode(latent)
    
    return images

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = DDMConfig()
    config.use_top_k = 1  # Use top-1 expert for efficiency
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb if requested
    if args.log_to_wandb:
        # Use a separate thread for wandb initialization to avoid blocking
        def init_wandb():
            wandb.init(
                project="ddm-inference",
                config={
                    "prompt": args.prompt,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "guidance_scale": 7.5,
                    "use_distilled": args.use_distilled,
                    "image_size": config.image_size,
                    "num_experts": config.num_experts
                },
                settings=wandb.Settings(start_method="thread", _disable_stats=True)
            )
            logger.info("Initialized wandb for logging")
        
        # Start wandb initialization in a separate thread
        init_thread = threading.Thread(target=init_wandb, daemon=True)
        init_thread.start()
        init_thread.join()  # Wait for initialization to complete
    
    # Load models
    if args.use_distilled:
        # Load distilled model
        distilled_path = os.path.join(args.checkpoint_dir, "distilled_model.pt")
        if not os.path.exists(distilled_path):
            raise FileNotFoundError(f"Distilled model not found at {distilled_path}")
        
        model = ExpertDiT(config).to(device)
        model.load_state_dict(torch.load(distilled_path))
        model.eval()
        
        # For distilled model, we use it as both router and expert
        router = model
        experts = [model]
        logger.info("Using distilled model for inference")
    else:
        # Load router
        router_path = os.path.join(args.checkpoint_dir, "router_step400000.pt")
        if not os.path.exists(router_path):
            # Try to find the latest router checkpoint
            router_files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith("router_step")]
            if not router_files:
                raise FileNotFoundError(f"No router checkpoints found in {args.checkpoint_dir}")
            router_path = os.path.join(args.checkpoint_dir, sorted(router_files)[-1])
        
        router = RouterModel(config).to(device)
        router.load_state_dict(torch.load(router_path))
        router.eval()
        
        # Load experts
        experts = []
        for i in range(config.num_experts):
            expert_path = os.path.join(args.checkpoint_dir, f"expert_{i}_step400000.pt")
            if not os.path.exists(expert_path):
                # Try to find the latest expert checkpoint
                expert_files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith(f"expert_{i}_step")]
                if not expert_files:
                    raise FileNotFoundError(f"No checkpoints found for expert {i} in {args.checkpoint_dir}")
                expert_path = os.path.join(args.checkpoint_dir, sorted(expert_files)[-1])
            
            expert = ExpertDiT(config).to(device)
            expert.load_state_dict(torch.load(expert_path))
            expert.eval()
            experts.append(expert)
        
        logger.info(f"Loaded router and {len(experts)} experts for inference")
    
    # Load VAE and CLIP
    vae_wrapper = VAEWrapper(device, config)
    clip_encoder = CLIPTextEncoder(device, config)
    
    # Set seed if provided for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Generate images
    logger.info(f"Generating images with prompt: '{args.prompt}'")
    images = ddm_sample(
        config,
        router,
        experts,
        vae_wrapper,
        clip_encoder.encode([args.prompt] * args.batch_size),
        num_steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        guidance_scale=7.5,
        uncond_embeddings=clip_encoder.encode([""] * args.batch_size),
        log_to_wandb=args.log_to_wandb
    )
    
    # Save images
    for i in range(args.batch_size):
        # Convert to numpy and adjust range
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np + 1.0) * 127.5  # Convert from [-1, 1] to [0, 255]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Save as PNG
        img_pil = Image.fromarray(img_np)
        img_path = os.path.join(args.output_dir, f"ddm_sample_{i}.png")
        img_pil.save(img_path)
        logger.info(f"Saved image to {img_path}")
    
    # Log images to wandb if requested
    if args.log_to_wandb and wandb.run is not None:
        wandb_images = []
        for i in range(args.batch_size):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
            wandb_images.append(wandb.Image(img_np, caption=f"Sample {i}: {args.prompt}"))
        
        # Use helper function for non-blocking logging
        log_to_wandb_async({"generated_images": wandb_images})
    
    # Clean up
    if args.log_to_wandb:
        wandb.finish()
    
    # Signal the logging thread to exit
    log_queue.put(None)
    log_thread.join()

if __name__ == "__main__":
    args = parse_args()
    main(args) 