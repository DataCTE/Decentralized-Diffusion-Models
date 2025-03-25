"""Inference for Decentralized Diffusion Models"""

import os
import torch
import torch.distributed as dist
import numpy as np
import datetime
import json
from PIL import Image
from queue import Queue
from threading import Thread

from config import DDMConfig
from models.mmdit import ExpertMMDiT
from models.router import RouterModel
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from utils.visualization import tensor_to_pil
from utils.distributed import is_main_process, get_rank, setup_distributed

# Import centralized utilities
from utils.logging import setup_logger, log_images, setup_distributed_logger
from trainers.sampling import ddm_sample, distilled_sample

# Initialize logger
logger = None

def log_worker(queue):
    """Background worker thread for logging generated images"""
    while True:
        item = queue.get()
        if item is None:  # Sentinel to stop the thread
            break
            
        try:
            log_to_wandb_async(item)
        except Exception as e:
            print(f"Error in logging worker: {str(e)}")
        finally:
            queue.task_done()

def log_to_wandb_async(data):
    """Log images to wandb without blocking inference"""
    import wandb
    
    if not wandb.run:
        return
        
    images = data.get('images', [])
    prompts = data.get('prompts', [])
    step = data.get('step', 0)
    
    # Convert tensors to PIL
    if isinstance(images, torch.Tensor):
        # Convert batch tensor to list of PIL images
        images = [tensor_to_pil(img) for img in images]
        
    # Create captions list if available
    captions = prompts if prompts else None
    
    # Log to wandb
    log_images(
        images=images,
        captions=captions,
        step=step,
        prefix="generated"
    )

def setup_environment(config):
    """Setup logging, directories, and environment"""
    global logger
    
    # Setup logging
    log_file = None
    if is_main_process():
        # Only create log files on main process
        os.makedirs(getattr(config, 'log_dir', 'logs'), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(getattr(config, 'log_dir', 'logs'), f"inference-{timestamp}.log")
    
    # Setup root logger
    logger = setup_distributed_logger("DDMInference", log_file=log_file, rank=get_rank())
    
    # Create output directories
    if is_main_process():
        os.makedirs(config.sample_dir, exist_ok=True)
        logger.info(f"Saving samples to: {config.sample_dir}")
    
    # Initialize wandb if configured
    if is_main_process() and getattr(config, 'use_wandb', False):
        from utils.logging import init_wandb
        run_name = getattr(config, 'wandb_run_name', None) or f"ddm-inference-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run = init_wandb(
            config=config,
            project=getattr(config, 'wandb_project', "decentralized-diffusion-inference"),
            name=run_name
        )
        logger.info(f"Initialized WandB logging: {run_name}")
        
        # Start background logging thread
        queue = Queue()
        log_thread = Thread(target=log_worker, args=(queue,))
        log_thread.daemon = True
        log_thread.start()
        return queue
    
    return None

def load_models(config, device, checkpoint_dir, cache_manager=None):
    """
    Load router and expert models
    
    Args:
        config: Configuration object
        device: Device to load models on
        checkpoint_dir: Directory containing checkpoints
        cache_manager: Optional ExpertCacheManager for efficient expert loading
        
    Returns:
        router_model: Router model
        expert_models: Dictionary of expert models
        vae: VAE model
        clip: CLIP model
    """
    # Load VAE model
    logger.info("Loading VAE model")
    vae = VAEWrapper(
        vae_model=config.vae_model,
        device=device
    )
    
    # Load CLIP model
    logger.info("Loading CLIP model")
    clip = CLIPTextEncoder(
        clip_model=config.clip_model,
        device=device
    )
    
    # Load router model using FSDP-aware loading
    router_model = RouterModel(
        config=config,
        num_experts=config.num_experts
    )  # No .to(device) - FSDP handles placement
    
    router_checkpoint = os.path.join(checkpoint_dir, "router_model.pt")
    if os.path.exists(router_checkpoint):
        load_models(
            router_model,
            path=router_checkpoint,
            is_fsdp=True,
            device=device
        )
    else:
        logger.warning(f"Router checkpoint {router_checkpoint} not found")
    
    # Create dictionary to hold expert models
    expert_models = {}
    
    # Load expert models (lazily if cache manager is provided)
    if cache_manager is None:
        # Without cache manager, load all experts at once
        for expert_idx in range(config.num_experts):
            expert_checkpoint = os.path.join(checkpoint_dir, f"expert_{expert_idx}.pt")
            if os.path.exists(expert_checkpoint):
                logger.info(f"Loading expert {expert_idx}")
                expert_model = ExpertMMDiT(config).to(device)
                state_dict = torch.load(expert_checkpoint, map_location=device)
                expert_model.load_state_dict(state_dict)
                expert_models[expert_idx] = expert_model
            else:
                logger.warning(f"Expert checkpoint {expert_checkpoint} not found")
    else:
        # With cache manager, just verify checkpoints exist and create loader functions
        for expert_idx in range(config.num_experts):
            expert_checkpoint = os.path.join(checkpoint_dir, f"expert_{expert_idx}.pt")
            if os.path.exists(expert_checkpoint):
                # Create a builder function for this expert
                def create_expert_builder(idx, checkpoint_path):
                    def builder(_):
                        logger.info(f"Loading expert {idx} with cache manager")
                        expert_model = ExpertMMDiT(config).to(device)
                        state_dict = torch.load(checkpoint_path, map_location=device)
                        expert_model.load_state_dict(state_dict)
                        return expert_model
                    return builder
                
                # Store the builder in the expert_models dictionary
                expert_models[expert_idx] = create_expert_builder(expert_idx, expert_checkpoint)
            else:
                logger.warning(f"Expert checkpoint {expert_checkpoint} not found")
    
    return router_model, expert_models, vae, clip

def load_distilled_model(config, device, checkpoint_path):
    """
    Load distilled model if available
    
    Args:
        config: Configuration object
        device: Device to load model on
        checkpoint_path: Path to distilled model checkpoint
        
    Returns:
        distilled_model: Distilled model or None if not available
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Distilled model checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        logger.info(f"Loading distilled model from {checkpoint_path}")
        distilled_model = ExpertMMDiT(config).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            distilled_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'ema_state_dict' in checkpoint:
            # Prefer EMA model if available
            distilled_model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            # Assume direct state dict
            distilled_model.load_state_dict(checkpoint)
            
        logger.info("Distilled model loaded successfully")
        return distilled_model
    except Exception as e:
        logger.error(f"Error loading distilled model: {str(e)}")
        return None

def load_prompts(prompts_file=None):
    """
    Load text prompts for inference
    
    Args:
        prompts_file: Optional path to JSON file containing prompts
        
    Returns:
        prompts: List of text prompts
    """
    default_prompts = [
        "A photo of a cat in a garden",
        "A painting of a mountain landscape",
        "A digital art of a futuristic city",
        "A photo of a red sports car"
    ]
    
    if prompts_file is None or not os.path.exists(prompts_file):
        logger.info("Using default text prompts")
        return default_prompts
    
    try:
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        if not isinstance(prompts, list):
            logger.warning("Prompts file does not contain a list, using default prompts")
            return default_prompts
            
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts file: {str(e)}")
        return default_prompts

def save_images(images, output_dir, prefix="sample"):
    """
    Save generated images to disk
    
    Args:
        images: List of PIL images
        output_dir: Directory to save images in
        prefix: Prefix for image filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        image.save(image_path)
    
    logger.info(f"Saved {len(images)} images to {output_dir}")

def get_expert_for_inference(expert_idx, expert_models, cache_manager=None):
    """
    Get expert model for inference, using cache manager if available
    
    Args:
        expert_idx: Expert index
        expert_models: Dictionary of expert models or builders
        cache_manager: Optional ExpertCacheManager
        
    Returns:
        expert_model: Expert model
    """
    if cache_manager is None:
        # Direct access
        return expert_models.get(expert_idx)
    else:
        # Use cache manager to retrieve expert
        if expert_idx not in expert_models:
            return None
            
        # Check if the value is a function (builder) or model
        if callable(expert_models[expert_idx]) and not isinstance(expert_models[expert_idx], torch.nn.Module):
            # It's a builder function
            builder = expert_models[expert_idx]
            return cache_manager.get_expert(expert_idx, builder)
        else:
            # It's already a model
            return expert_models[expert_idx]

def run_inference_pipeline(
    config,
    device,
    checkpoint_dir,
    output_dir,
    prompts_file=None,
    images_file=None,
    batch_size=4,
    num_steps=50,
    cache_manager=None
):
    """Run inference pipeline for Decentralized Diffusion Models"""
    # Add config validation
    if not config.enable_sampling:
        logger.error("Sampling disabled in config, aborting inference")
        return
    
    # Load models
    router_model, expert_models, vae, clip = load_models(config, device, checkpoint_dir, cache_manager)
    
    # Add expert count check
    if len(expert_models) == 0:
        logger.error("No experts found for inference")
        return
    
    # Add device synchronization
    torch.cuda.synchronize(device=device)
    
    # Load distilled model if available
    distilled_model = None
    distilled_path = os.path.join(checkpoint_dir, "distilled_model_best.pt")
    if os.path.exists(distilled_path):
        distilled_model = load_distilled_model(config, device, distilled_path)
    
    # Load prompts
    prompts = load_prompts(prompts_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process prompts in batches
    for batch_idx in range(0, len(prompts), batch_size):
        # Get batch of prompts
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        actual_batch_size = len(batch_prompts)
        
        # Encode prompts with CLIP
        logger.info(f"Encoding prompts batch {batch_idx//batch_size + 1}")
        text_embeddings = []
        for prompt in batch_prompts:
            embedding = clip.encode_text(prompt)
            text_embeddings.append(embedding)
        
        text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # Create unconditional embeddings for classifier-free guidance
        uncond_embeddings = clip.encode_text([""] * actual_batch_size)
        
        # Sample with DDM
        logger.info(f"Generating images with DDM for batch {batch_idx//batch_size + 1}")
        
        # Define latent shape based on VAE
        latent_shape = (actual_batch_size, config.latent_channels, 
                        config.image_size // 8, config.image_size // 8)
        
        try:
            # First try using distilled model if available
            if distilled_model is not None:
                logger.info("Using distilled model for sampling")
                latents = distilled_sample(
                    distilled_model=distilled_model,
                    shape=latent_shape,
                    num_steps=num_steps,
                    prompt_embeds=text_embeddings,
                    cfg_scale=config.cfg_scale,
                    device=device
                )
            else:
                # If no distilled model, use DDM sampling
                logger.info("Using DDM for sampling")
                # Load any experts to prefetch into cache if using cache manager
                if cache_manager is not None:
                    # Prefetch a few experts to optimize start time
                    for idx in range(min(3, config.num_experts)):
                        if idx in expert_models and callable(expert_models[idx]):
                            cache_manager.queue_prefetch(idx, expert_models[idx])
                
                # Get sampling parameters from config
                inference_strategy = getattr(config, 'inference_strategy', 'top_k')
                top_k = min(getattr(config, 'top_k', 1), len(expert_models))
                top_p = getattr(config, 'top_p', 0.9)
                true_clusters = None  # Oracle strategy not supported in inference.py
                
                latents = ddm_sample(
                    router=router_model,
                    experts=expert_models,
                    shape=latent_shape,
                    num_steps=num_steps,
                    device=device,
                    cfg_scale=config.cfg_scale,
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    inference_strategy=inference_strategy,
                    top_k=top_k,
                    top_p=top_p,
                    true_clusters=true_clusters
                )
            
            # Decode latents to images
            logger.info("Decoding latents to images")
            images = vae.decode_latents(latents)
            
            # Convert to PIL images
            pil_images = []
            for image in images:
                # Convert to PIL
                pil_image = Image.fromarray(
                    (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )
                pil_images.append(pil_image)
            
            # Save images
            batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx//batch_size + 1}")
            save_images(pil_images, batch_output_dir)
            
            # Save prompts
            with open(os.path.join(batch_output_dir, "prompts.json"), 'w') as f:
                prompts_dict = {i: prompt for i, prompt in enumerate(batch_prompts)}
                json.dump(prompts_dict, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            continue
    
    logger.info("Inference complete")

def main():
    # Load configuration
    config = DDMConfig()
    
    # Initialize distributed setup if needed
    if getattr(config, 'use_distributed', torch.cuda.device_count() > 1):
        setup_distributed(config)
    
    # Setup environment and get logging queue
    log_queue = setup_environment(config)
    
    try:
        # Run inference
        run_inference_pipeline(
            config=config,
            device=torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu"),
            checkpoint_dir="checkpoints",
            output_dir="samples",
            prompts_file=None,
            images_file=None,
            batch_size=4,
            num_steps=50,
            cache_manager=None
        )
        
        if is_main_process():
            logger.info(f"All samples saved to: samples")
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
    finally:
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()
            
        if is_main_process() and getattr(config, 'use_wandb', False):
            import wandb
            wandb.finish()

if __name__ == "__main__":
    main() 