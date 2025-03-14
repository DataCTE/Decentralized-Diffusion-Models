"""Inference script for sampling from trained Decentralized Diffusion Models."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import datetime
import time
import logging
from PIL import Image
from queue import Queue
from threading import Thread
import glob

from config import DDMConfig
from trainers.coordinator import DDMTrainingCoordinator
from utils.checkpoint import load_model_checkpoint
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder
from utils.visualization import tensor_to_pil, create_image_grid
from utils.distributed import is_main_process, get_rank, get_world_size, setup_distributed

# Import centralized utilities
from utils.logging import setup_logger, log_metrics, log_images
from utils.sampling import ddm_sample

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
    logger = setup_logger("DDMInference", rank=get_rank(), log_file=log_file)
    
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

def load_models(config, device):
    """Load models needed for inference"""
    logger.info("Loading models...")
    
    # Load VAE
    vae = VAEWrapper(device, config)
    logger.info("VAE loaded successfully")
    
    # Load CLIP
    clip = CLIPTextEncoder(device, config)
    logger.info("CLIP loaded successfully")
    
    # Find the latest checkpoint if not specified
    if not hasattr(config, 'checkpoint_path') or not config.checkpoint_path:
        logger.info("Checkpoint path not specified, looking for latest checkpoint...")
        checkpoints = glob.glob(os.path.join(config.checkpoint_dir, "*.pt"))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {config.checkpoint_dir}")
        # Get the most recent checkpoint
        config.checkpoint_path = max(checkpoints, key=os.path.getctime)
    
    logger.info(f"Using checkpoint: {config.checkpoint_path}")
    
    # Load the model from checkpoint
    from models.dit import ExpertDiT
    from models.router import RouterModel
    
    # Initialize models
    router = RouterModel(config).to(device)
    experts = []
    
    if hasattr(config, 'use_distilled_model') and config.use_distilled_model:
        # Load distilled model
        logger.info("Loading distilled model...")
        distilled_model = ExpertDiT(config).to(device)
        load_model_checkpoint(distilled_model, config.distilled_model_path, device=device)
        logger.info("Distilled model loaded successfully")
        return vae, clip, router, [distilled_model]
    else:
        # Load router and experts
        logger.info("Loading router model...")
        load_model_checkpoint(router, config.checkpoint_path, device=device)
        logger.info("Router loaded successfully")
        
        # Load experts
        num_experts = getattr(config, 'num_experts', 8)
        expert_paths = []
        
        # Find expert checkpoints
        if hasattr(config, 'expert_paths') and config.expert_paths:
            expert_paths = config.expert_paths
        else:
            # Look for expert checkpoints in the same directory
            checkpoint_dir = os.path.dirname(config.checkpoint_path)
            for i in range(num_experts):
                expert_path = os.path.join(checkpoint_dir, f"expert_{i}_*.pt")
                matches = glob.glob(expert_path)
                if matches:
                    expert_paths.append(max(matches, key=os.path.getctime))
                else:
                    logger.warning(f"No checkpoint found for expert {i}")
        
        # Load each expert
        for path in expert_paths:
            expert = ExpertDiT(config).to(device)
            load_model_checkpoint(expert, path, device=device)
            experts.append(expert)
            
        logger.info(f"Loaded {len(experts)} experts successfully")
        return vae, clip, router, experts

def run_inference(config, log_queue=None):
    """Run inference using loaded models"""
    # Set device
    device = torch.device(f"cuda:{get_rank()}" if torch.cuda.is_available() else "cpu")
    
    # Load models
    vae, clip, router, experts = load_models(config, device)
    
    # Get inference parameters from config
    batch_size = getattr(config, 'inference_batch_size', 1)
    steps = getattr(config, 'inference_steps', 50)
    guidance_scale = getattr(config, 'cfg_scale', 7.5)
    top_k = getattr(config, 'inference_top_k', 1)
    seed = getattr(config, 'seed', int(time.time()))
    
    # Get prompts from config
    if hasattr(config, 'inference_prompts') and config.inference_prompts:
        prompts = config.inference_prompts
        if isinstance(prompts, str):
            prompts = [prompts]
    else:
        prompts = ["a photo of a cat"]
    
    # Number of samples per prompt
    num_samples = getattr(config, 'num_samples_per_prompt', 1)
    total_samples = len(prompts) * num_samples
    
    # Log inference settings
    logger.info(f"Running inference with:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Steps: {steps}")
    logger.info(f"  - CFG scale: {guidance_scale}")
    logger.info(f"  - Top-k experts: {top_k}")
    logger.info(f"  - Seed: {seed}")
    logger.info(f"  - Prompts: {prompts}")
    logger.info(f"  - Samples per prompt: {num_samples}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(config.sample_dir, f"samples_{timestamp}")
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
    
    # Inference loop
    start_time = time.time()
    sample_count = 0
    
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"Generating samples for prompt [{prompt_idx+1}/{len(prompts)}]: '{prompt}'")
        
        # Encode prompt with CLIP
        text_embeddings, uncond_embeddings = clip.encode_with_uncond([prompt])
        
        for sample_idx in range(num_samples):
            # Get latent shape
            sample_shape = (
                batch_size,
                config.latent_channels,
                config.image_size // 8,
                config.image_size // 8
            )
            
            # Sample from model
            try:
                logger.info(f"Sampling batch {sample_idx+1}/{num_samples} for prompt: '{prompt}'")
                batch_start = time.time()
                
                latents = ddm_sample(
                    router=router,
                    experts=experts,
                    shape=sample_shape,
                    steps=steps,
                    top_k=top_k,
                    device=device,
                    cfg_scale=guidance_scale,
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings
                )
                
                # Decode latents
                logger.info("Decoding latents...")
                images = vae.decode(latents)
                
                # Convert to PIL images
                pil_images = [tensor_to_pil(img) for img in images]
                
                # Save individual images
                if is_main_process():
                    for i, img in enumerate(pil_images):
                        # Create a filename with prompt info
                        prompt_tag = prompt.replace(" ", "_").replace(",", "").replace(".", "")
                        prompt_tag = "".join(c for c in prompt_tag if c.isalnum() or c == '_')
                        prompt_tag = prompt_tag[:50]  # Limit length
                        
                        filename = f"{prompt_tag}_{sample_idx}_{i}.png"
                        img_path = os.path.join(output_dir, filename)
                        img.save(img_path)
                        logger.info(f"Saved image to {img_path}")
                        
                        sample_count += 1
                
                # Log to wandb if configured
                if log_queue is not None and is_main_process():
                    log_queue.put({
                        'images': pil_images,
                        'prompts': [prompt] * len(pil_images),
                        'step': sample_count
                    })
                
                # Create and save a grid for this batch
                if batch_size > 1 and is_main_process():
                    grid = create_image_grid(pil_images)
                    grid_filename = f"{prompt_tag}_grid_{sample_idx}.png"
                    grid_path = os.path.join(output_dir, grid_filename)
                    grid.save(grid_path)
                    logger.info(f"Saved grid to {grid_path}")
                
                batch_time = time.time() - batch_start
                logger.info(f"Batch completed in {batch_time:.2f}s ({batch_time / batch_size:.2f}s per image)")
                
            except Exception as e:
                logger.error(f"Error during sampling: {str(e)}", exc_info=True)
    
    # Log completion statistics
    total_time = time.time() - start_time
    logger.info(f"Inference completed: {sample_count} samples generated in {total_time:.2f}s")
    logger.info(f"Average time per sample: {total_time / max(1, sample_count):.2f}s")
    
    # Signal the logging thread to exit
    if log_queue is not None and is_main_process():
        log_queue.put(None)
        log_queue.join()  # Wait for queue to empty
    
    return output_dir

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
        output_dir = run_inference(config, log_queue)
        
        if is_main_process():
            logger.info(f"All samples saved to: {output_dir}")
            
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