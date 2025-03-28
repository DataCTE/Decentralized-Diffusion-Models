"""Inference for Decentralized Diffusion Models"""

import os
import torch
import torch.distributed as dist
import datetime
import json
import logging # Import standard logging
import tqdm
from queue import Queue
from threading import Thread

# Import necessary components from the project
# from config import DDMConfig # Assuming DDMConfig might be replaced by get_config
from config import get_config
from models.mmdit import ExpertMMDiT # Keep ExpertMMDiT import
from models.router import RouterModel
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder # Keep if used for encoding prompts
from utils.visualization import tensor_to_pil
from utils.distributed import is_main_process, get_rank, get_world_size, setup_distributed, is_dist_initialized
from utils.logging import setup_logger, log_images # Keep setup_logger, log_images
from types import SimpleNamespace
from trainers.sampling import ddm_sample, distilled_sample
from utils.checkpoint import load_model_checkpoint # Assuming a generic loader exists
from utils.expert_cache import ExpertCacheManager # Assuming cache manager is used
from models.mmdit import FluxParams # Import if needed by ExpertMMDiT

# Define the global logger, initialized to None
logger = None

def log_worker(queue):
    """Background worker thread for logging generated images"""
    # This function runs in a separate thread. Direct logging using the global logger
    # might be problematic. Using print for internal errors is safer here,
    # or implement a thread-safe logging queue if needed.
    while True:
        item = queue.get()
        if item is None:  # Sentinel to stop the thread
            break
        try:
            log_to_wandb_async(item)
        except Exception as e:
            print(f"Error in logging worker: {str(e)}") # Keep print for thread safety
        finally:
            queue.task_done()

def log_to_wandb_async(data):
    """Logs data to WandB asynchronously (called by worker)."""
    # This function also runs in the worker thread.
    try:
        import wandb
        if wandb.run:
            # Assuming log_images or wandb.log is thread-safe enough for this use case.
            # If issues arise, consider passing data back to the main thread for logging.
            # Example: log wandb.Image directly if data contains PIL images
            step = data.get('step', None)
            log_payload = {k: v for k, v in data.items() if k not in ['step']} # Prepare data for wandb.log
            wandb.log(log_payload, step=step)
        # else:
        #    print("WandB run not active, cannot log data.")
    except ImportError:
        # print("WandB not installed, cannot log data.")
        pass
    except Exception as e:
        print(f"Error during WandB logging: {str(e)}") # Keep print for thread safety

# Modify setup_environment to initialize the *global* logger
def setup_environment(config):
    """Setup logging, directories, distributed env, and environment variables"""
    global logger # Declare intention to modify the global logger

    rank = 0
    world_size = 1
    log_file = None

    # Attempt distributed setup first if configured or multiple GPUs detected
    use_distributed = getattr(config, 'distributed_inference', torch.cuda.device_count() > 1)
    if use_distributed:
        try:
            # setup_distributed should return rank and world_size
            rank, world_size = setup_distributed()
        except Exception as e:
            # Log error using a temporary basic logger before the main one is set up
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Failed to setup distributed environment: {e}", exc_info=True)
            # Decide whether to raise or fallback to single process
            raise RuntimeError("Distributed setup failed") from e
    else:
        # Ensure CUDA device is set even in non-distributed mode if available
        if torch.cuda.is_available():
            torch.cuda.set_device(0)


    # Setup logger AFTER determining rank and world_size
    if is_main_process(rank): # Use rank directly
        # Only create log files and directories on main process
        log_dir = getattr(config, 'log_dir', os.path.join(config.output_dir, 'logs'))
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"inference-{timestamp}-rank{rank}.log")

        # Also create sample directory here
        sample_dir = getattr(config, 'sample_dir', os.path.join(config.output_dir, 'samples'))
        os.makedirs(sample_dir, exist_ok=True)

    # Use the centralized setup_logger function to initialize the global logger
    logger = setup_logger(
        name="DDMInference", # Consistent logger name
        output_dir=config.output_dir, # Pass output_dir for potential file logging path
        log_to_console=True, # Keep console logging for inference
        level=logging.DEBUG if getattr(config, 'verbose_logging', False) else logging.INFO, # Control level via config
        rank=rank,
        world_size=world_size
        # log_file=log_file # setup_logger can handle file path creation based on output_dir and rank
    )

    logger.info(f"Environment setup complete. Rank: {rank}, World Size: {world_size}")
    if is_main_process(rank):
        logger.info(f"Logging to console and potentially to file in {config.output_dir}/logs")
        sample_dir = getattr(config, 'sample_dir', os.path.join(config.output_dir, 'samples'))
        logger.info(f"Saving samples to: {sample_dir}")


    # Initialize wandb if configured (only on main process)
    log_queue = None
    if is_main_process(rank) and getattr(config, 'wandb_enabled', False):
        try:
            import wandb
            # Check if already initialized (e.g., in testing scenarios)
            if wandb.run is None:
                run_name = getattr(config, 'wandb_run_name', f"ddm-inference-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
                wandb.init(
                    project=getattr(config, 'wandb_project', "decentralized-diffusion-inference"),
                    config=vars(config), # Log the config
                    name=run_name,
                    dir=getattr(config, 'wandb_dir', config.output_dir), # Set wandb dir
                    settings=wandb.Settings(start_method="thread") # Use thread start method
                )
                logger.info(f"Initialized WandB logging: {run_name} (URL: {wandb.run.get_url()})")
            else:
                 logger.warning("WandB run already initialized.")

            # Start background logging thread
            log_queue = Queue()
            log_thread = Thread(target=log_worker, args=(log_queue,), daemon=True)
            log_thread.start()
            logger.info("Started background WandB logging thread.")
        except ImportError:
             logger.error("WandB is enabled in config, but 'wandb' package is not installed. Disabling WandB logging.")
             config.wandb_enabled = False # Ensure it's disabled if import fails
        except Exception as e:
             logger.error(f"Failed to initialize WandB: {e}", exc_info=True)
             config.wandb_enabled = False

    # Return rank, world_size, device, and log_queue
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device, log_queue


# Ensure load_models uses the global logger
def load_models(config, device, checkpoint_dir, cache_manager=None):
    """Loads the router, experts (via cache/builder), VAE, and CLIP."""
    global logger # Access the global logger

    logger.info(f"Loading models. Checkpoint dir: {checkpoint_dir}")

    # --- Find latest checkpoint (common logic for router/experts) ---
    # This assumes checkpoints are saved with a pattern like ddm_step_XXXX.pt
    # If router and experts are saved separately, adjust finding logic.
    latest_checkpoint_path = None
    step = 0
    if os.path.isdir(checkpoint_dir):
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and f.startswith("ddm_step_")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                step = int(checkpoints[-1].split('_')[-1].split('.')[0])
                logger.info(f"Found latest DDM checkpoint: {latest_checkpoint_path} (Step: {step})")
            else:
                 logger.warning(f"No DDM checkpoints (ddm_step_*.pt) found in {checkpoint_dir}. Models might not be loaded.")
        except Exception as e:
             logger.error(f"Error finding latest checkpoint in {checkpoint_dir}: {e}", exc_info=True)
             # Decide if this is critical
             # raise RuntimeError("Failed to find valid checkpoint.") from e
    else:
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")


    # --- Load VAE ---
    try:
        logger.info("Loading VAE...")
        vae = VAEWrapper(device, config)
        _ = vae.vae # Trigger lazy loading
        logger.info("VAE loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load VAE.")
        raise RuntimeError("VAE loading failed, cannot proceed.") from e

    # --- Load CLIP ---
    try:
        logger.info("Loading CLIP...")
        # Assuming CLIPTextEncoder takes device and config
        clip = CLIPTextEncoder(device, config)
        logger.info("CLIP loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load CLIP.")
        raise RuntimeError("CLIP loading failed, cannot proceed.") from e


    # --- Load Router ---
    # Initialize router (FSDP wrapping happens inside wrap_model_with_fsdp if needed)
    logger.info("Initializing Router model...")
    router_model = RouterModel(config) # No .to(device) yet if using FSDP loading

    # Use centralized checkpoint loading function
    if latest_checkpoint_path:
        logger.info(f"Loading Router state from {latest_checkpoint_path}...")
        try:
            # Assuming load_model_checkpoint handles FSDP state loading correctly
            router_metadata = load_model_checkpoint(
                model=router_model,
                path=latest_checkpoint_path,
                model_key='router_model_state', # Key in the checkpoint file
                device=device, # Target device
                is_fsdp=getattr(config, 'use_fsdp', False), # Check if FSDP was used
                rank=get_rank(),
                world_size=get_world_size()
            )
            if router_metadata:
                logger.info(f"Router loaded successfully (Step: {router_metadata.get('step', 'N/A')}).")
            else:
                 logger.warning(f"Router state not found or failed to load from {latest_checkpoint_path}.")
        except Exception as e:
             logger.error(f"Error loading router state from checkpoint: {e}", exc_info=True)
             # Decide if loading failure is critical
    else:
        logger.warning("No checkpoint found, router weights are uninitialized.")


    # --- Setup Expert Loading ---
    # We don't load all experts at once. Instead, create builders or use cache manager.
    expert_models = {} # Dictionary to hold models or builders

    def create_expert_loader(expert_idx):
        """Factory function to load a single expert model on demand."""
        global logger # Access global logger
        logger.info(f"Factory: Creating/Loading Expert {expert_idx}...")
        # Initialize the expert model structure
        # Important: Ensure FluxParams or similar config is correctly passed if needed
        expert_cfg_ns = SimpleNamespace(**vars(config)) # Create namespace if needed
        expert_cfg_ns.in_channels = config.latent_channels # Ensure correct channels

        # Use FluxParams dataclass if ExpertMMDiT expects it
        try:
             flux_params = FluxParams(**vars(expert_cfg_ns)) # Adapt based on FluxParams definition
             base_expert = ExpertMMDiT(flux_params)
        except TypeError as te:
             logger.error(f"TypeError initializing ExpertMMDiT {expert_idx}. Check config/FluxParams: {te}")
             # Fallback or re-raise
             base_expert = ExpertMMDiT(expert_cfg_ns) # Try with namespace if dataclass fails

        # Load state dict from the main checkpoint file if found
        if latest_checkpoint_path:
            logger.debug(f"Factory: Loading Expert {expert_idx} state from {latest_checkpoint_path}...")
            try:
                expert_metadata = load_model_checkpoint(
                    model=base_expert,
                    path=latest_checkpoint_path,
                    model_key=f'expert_models_state.{expert_idx}', # Key for this specific expert
                    device=device, # Load directly to target device
                    is_fsdp=getattr(config, 'use_fsdp', False),
                    rank=get_rank(),
                    world_size=get_world_size()
                )
                if expert_metadata:
                     logger.debug(f"Factory: Expert {expert_idx} loaded successfully.")
                else:
                     logger.warning(f"Factory: Expert {expert_idx} state not found or failed to load from checkpoint.")
            except Exception as e:
                 logger.error(f"Factory: Error loading expert {expert_idx} state: {e}", exc_info=True)
                 # Model remains uninitialized if loading fails
        else:
             logger.warning(f"Factory: No checkpoint for Expert {expert_idx}, weights uninitialized.")

        # FSDP Wrapping should happen *outside* the factory if using cache manager,
        # or here if loading directly without cache manager but using FSDP.
        # Let's assume cache manager handles FSDP wrapping for now.
        return base_expert.to(device).eval() # Ensure eval mode and correct device

    # Populate the expert_models dictionary with loader functions
    for i in range(config.num_experts):
        expert_models[i] = create_expert_loader # Store the factory function

    logger.info(f"Setup complete for loading {config.num_experts} experts on demand.")

    return router_model.to(device).eval(), expert_models, vae, clip # Ensure router is on device and eval mode

# Ensure load_distilled_model uses the global logger
def load_distilled_model(config, device, checkpoint_path):
    """Load distilled model if available"""
    global logger
    # ... (rest of the function remains the same, using logger) ...

# Ensure load_prompts uses the global logger
def load_prompts(prompts_file=None):
    """Load text prompts for inference"""
    global logger
    # ... (rest of the function remains the same, using logger) ...

# Ensure save_images uses the global logger AND returns paths
def save_images(images, output_dir, prefix="sample"):
    """Save generated images to disk and return their paths."""
    global logger
    saved_paths = [] # List to store paths of saved images
    try:
        os.makedirs(output_dir, exist_ok=True)
        for i, image in enumerate(images):
            # Use 8-digit padding for step consistency if prefix includes step
            image_filename = f"{prefix}_{i:04d}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            saved_paths.append(image_path) # Store the path

        if saved_paths: # Log only if images were actually saved
             logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        else:
             logger.warning(f"No images were provided to save_images for dir: {output_dir}")

    except Exception as e:
         logger.error(f"Error saving images to {output_dir}: {e}", exc_info=True)
         # Depending on severity, you might want to re-raise or just return empty list

    return saved_paths # Return the list of paths

# Ensure get_expert_for_inference uses the global logger (if it needs logging)
def get_expert_for_inference(expert_idx, expert_models, cache_manager=None):
    """Get expert model for inference, using cache manager if available"""
    global logger # Access global logger if needed for logging inside this func

    if cache_manager is None:
        # Direct access: expert_models dictionary holds builders or pre-loaded models
        if expert_idx not in expert_models:
            logger.error(f"Expert index {expert_idx} not found in expert_models dictionary.")
            return None
        expert_source = expert_models[expert_idx]
        if callable(expert_source) and not isinstance(expert_source, torch.nn.Module):
            # It's a builder function, call it to get the model
            logger.debug(f"Building expert {expert_idx} directly (no cache).")
            try:
                return expert_source(expert_idx) # Call the factory
            except Exception as e:
                logger.error(f"Error building expert {expert_idx} directly: {e}", exc_info=True)
                return None
        else:
            # Assume it's already a loaded model
            return expert_source
    else:
        # Use cache manager to retrieve/build expert
        if expert_idx not in expert_models:
             logger.error(f"Expert index {expert_idx} not found in expert_models dictionary (required for cache manager).")
             return None

        builder = expert_models[expert_idx] # Get the builder function
        if not callable(builder):
             logger.error(f"Source for expert {expert_idx} is not a callable builder function (required for cache manager).")
             return None # Cache manager needs the builder function

        # Cache manager handles calling the builder, FSDP wrapping (if configured), device placement etc.
        try:
            # Pass the builder function for the specific expert index
            return cache_manager.get_expert(expert_idx, lambda idx=expert_idx: builder(idx))
        except Exception as e:
             logger.error(f"Error getting expert {expert_idx} via cache manager: {e}", exc_info=True)
             return None


# --- run_inference_pipeline ---
# This function should REMAIN UNCHANGED as per the request.
# It will use the global 'logger' initialized by setup_environment.
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
    global logger # Indicate usage of global logger

    # Add config validation
    if not getattr(config, 'enable_sampling', True): # Default to True if not set
        logger.error("Sampling disabled in config (enable_sampling=False), aborting inference")
        return

    # Load models (uses global logger internally)
    # Pass cache_manager if provided
    router_model, expert_models_or_builders, vae, clip = load_models(config, device, checkpoint_dir, cache_manager)

    # Add expert count check
    num_experts_available = len(expert_models_or_builders)
    if num_experts_available == 0:
        logger.error("No experts found or loaded for inference.")
        return
    elif num_experts_available != config.num_experts:
         logger.warning(f"Mismatch between configured num_experts ({config.num_experts}) and available experts ({num_experts_available}).")


    # Add device synchronization (good practice before heavy compute)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)

    # Load distilled model if available (uses global logger internally)
    distilled_model = None
    distilled_path = os.path.join(checkpoint_dir, "distilled_model_best.pt") # Assuming standard name
    if os.path.exists(distilled_path):
        distilled_model = load_distilled_model(config, device, distilled_path)

    # Load prompts (uses global logger internally)
    prompts = load_prompts(prompts_file)
    if not prompts:
         logger.error("No prompts available for inference.")
         return # Exit if no prompts

    # Create output directory (already handled in setup_environment if rank 0)
    # os.makedirs(output_dir, exist_ok=True) # Redundant if setup_environment ran

    # --- Inference Loop ---
    # Use rank from distributed setup for progress bar disabling
    rank = get_rank()
    inference_pbar = tqdm(
        range(0, len(prompts), batch_size),
        desc="Generating Samples",
        disable=not is_main_process(rank) # Disable if not rank 0
    )

    all_generated_image_paths = [] # Collect paths if images_file is specified

    for batch_start_idx in inference_pbar:
        batch_end_idx = min(batch_start_idx + batch_size, len(prompts))
        batch_prompts = prompts[batch_start_idx:batch_end_idx]
        actual_batch_size = len(batch_prompts)

        if actual_batch_size == 0:
            continue

        batch_num = batch_start_idx // batch_size + 1
        logger.info(f"Processing Batch {batch_num}: {actual_batch_size} prompts...")

        # Encode prompts with CLIP
        try:
            # Assuming clip.encode returns a tensor ready for use
            text_embeddings = clip.encode(batch_prompts).to(device)
            # Create unconditional embeddings for classifier-free guidance
            uncond_embeddings = clip.encode([""] * actual_batch_size).to(device)
            logger.debug(f"Batch {batch_num}: Text embeddings shape: {text_embeddings.shape}")
        except Exception as e:
            logger.error(f"Error encoding prompts for batch {batch_num}: {e}", exc_info=True)
            continue # Skip batch if encoding fails

        # Define latent shape based on VAE and config
        # Ensure config has image_size or buckets defined
        h_pixels, w_pixels = config.image_size if hasattr(config, 'image_size') else config.buckets[0]
        latent_h, latent_w = h_pixels // 8, w_pixels // 8
        latent_shape = (actual_batch_size, config.latent_channels, latent_h, latent_w)
        logger.debug(f"Batch {batch_num}: Latent shape: {latent_shape}")


        try:
            latents = None
            # First try using distilled model if available
            if distilled_model is not None:
                logger.info(f"Batch {batch_num}: Using distilled model for sampling...")
                distilled_model.eval() # Ensure eval mode
                latents = distilled_sample(
                    distilled_model=distilled_model,
                    shape=latent_shape,
                    num_steps=num_steps,
                    prompt_embeds=text_embeddings, # Pass correct embedding key
                    cfg_scale=config.cfg_scale,
                    device=device
                    # Add other distilled_sample params as needed
                )
            else:
                # If no distilled model, use DDM sampling
                logger.info(f"Batch {batch_num}: Using DDM sampling...")

                # Router and experts should be in eval mode
                router_model.eval()
                # Note: Expert eval mode might be handled by cache manager or get_expert_for_inference

                # Get sampling parameters from config
                inference_strategy = getattr(config, 'inference_strategy', 'top_k')
                # Ensure top_k doesn't exceed available experts
                top_k = min(getattr(config, 'top_k', 1), num_experts_available)
                top_p = getattr(config, 'top_p', 0.9)

                # We pass the dictionary of expert loaders/models to ddm_sample.
                # ddm_sample will need to call get_expert_for_inference internally.
                latents = ddm_sample(
                    router=router_model,
                    # Pass the dictionary containing builders or potentially cached models
                    experts=expert_models_or_builders,
                    shape=latent_shape,
                    num_steps=num_steps,
                    device=device,
                    cfg_scale=config.cfg_scale,
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    inference_strategy=inference_strategy,
                    top_k=top_k,
                    top_p=top_p,
                    cache_manager=cache_manager, # Pass cache_manager to ddm_sample
                    verbose=(rank==0), # Only verbose on rank 0
                    # Ensure ddm_sample handles calling get_expert_for_inference
                )

            if latents is None:
                 logger.error(f"Batch {batch_num}: Sampling failed to produce latents.")
                 continue

            logger.info(f"Batch {batch_num}: Sampling complete. Decoding latents...")
            # Decode latents to images
            images_tensor = vae.decode(latents.to(vae.precision)) # Ensure correct dtype for VAE
            images_pil = tensor_to_pil(images_tensor) # Convert tensor [B, C, H, W] range [-1, 1] to list of PIL
            logger.info(f"Batch {batch_num}: Decoding complete ({len(images_pil)} images).")

            # Save images and prompts (only on rank 0)
            if is_main_process(rank):
                batch_output_dir = os.path.join(output_dir, f"batch_{batch_num:04d}")
                logger.info(f"Batch {batch_num}: Saving images to {batch_output_dir}...")
                # Call save_images and collect the returned paths
                saved_paths = save_images(images_pil, batch_output_dir, prefix=f"sample_batch_{batch_num}")
                if saved_paths: # Check if saving was successful
                    all_generated_image_paths.extend(saved_paths)
                    logger.info(f"Batch {batch_num}: Successfully saved {len(saved_paths)} images.")
                else:
                    logger.error(f"Batch {batch_num}: Failed to save images.")

                # Save corresponding prompts for the batch
                try:
                    with open(os.path.join(batch_output_dir, "prompts.json"), 'w') as f:
                        # Save as a list or dictionary
                        prompts_data = {idx: prompt for idx, prompt in enumerate(batch_prompts)}
                        json.dump(prompts_data, f, indent=2)
                except Exception as json_e:
                    logger.error(f"Failed to save prompts for batch {batch_num}: {json_e}")

                # Log images to WandB via queue if enabled
                # Make sure log_queue exists (created in setup_environment)
                if config.wandb_enabled and 'log_queue' in locals() and log_queue is not None:
                    wandb_data = {
                         # Log PIL images directly
                         f"inference/batch_{batch_num}_sample_{j}": wandb.Image(img, caption=batch_prompts[j])
                         for j, img in enumerate(images_pil)
                    }
                    log_queue.put({"step": batch_num, **wandb_data}) # Add step info for wandb logging


        except Exception as e:
            logger.exception(f"Error processing batch {batch_num}: {e}") # Log full traceback
            continue # Continue to the next batch

    inference_pbar.close()
    logger.info("Inference loop finished.")

    # Write image paths to file if requested (Rank 0 only)
    if is_main_process(rank) and images_file:
        logger.info(f"Saving list of generated image paths to {images_file}...")
        try:
            output_dir_for_paths = os.path.dirname(images_file)
            if output_dir_for_paths:
                 os.makedirs(output_dir_for_paths, exist_ok=True)
            with open(images_file, 'w') as f:
                for path in all_generated_image_paths:
                    f.write(path + '\n')
            logger.info(f"Image paths saved successfully.")
        except Exception as e:
            logger.error(f"Error writing image paths to {images_file}: {e}")



# --- main function ---
def main():
    global logger # Access global logger

    # Load configuration
    try:
        config = get_config("config.py") # Assumes config.py and get_config work
        # Add attributes if they are missing from config but needed by inference
        if not hasattr(config, 'output_dir'): config.output_dir = './inference_output'
        if not hasattr(config, 'distributed_inference'): config.distributed_inference = torch.cuda.device_count() > 1
        if not hasattr(config, 'verbose_logging'): config.verbose_logging = False
        if not hasattr(config, 'wandb_enabled'): config.wandb_enabled = False
        if not hasattr(config, 'enable_sampling'): config.enable_sampling = True # Inference needs sampling

    except FileNotFoundError:
        print("ERROR: config.py not found. Please create a configuration file.")
        return 1 # Indicate error
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1

    rank = 0
    world_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_queue = None


    try:
        # Setup environment (handles distributed setup and initializes global logger)
        rank, world_size, device, log_queue = setup_environment(config) # log_queue might be None

        # --- Initialize Cache Manager (Optional) ---
        cache_manager = None
        if getattr(config, 'use_expert_cache', False): # Check if caching is enabled
            logger.info("Initializing Expert Cache Manager...")
            try:
                cache_manager = ExpertCacheManager(
                    config=config,
                    device=device,
                    # max_experts, cpu_offload taken from config inside ExpertCacheManager
                    logger=logger # Pass the initialized logger
                )
            except Exception as e:
                 logger.error(f"Failed to initialize ExpertCacheManager: {e}", exc_info=True)
                 # Decide if this is fatal or continue without cache
                 logger.warning("Continuing without expert caching.")


        # --- Run Inference ---
        # Checkpoint directory logic
        checkpoint_dir = getattr(config, 'checkpoint_dir', os.path.join(config.output_dir, 'checkpoints'))
        if not os.path.isdir(checkpoint_dir):
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        output_dir = getattr(config, 'sample_dir', os.path.join(config.output_dir, 'inference_output'))

        run_inference_pipeline(
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            prompts_file=getattr(config, 'inference_prompts_file', None),
            images_file=getattr(config, 'inference_output_paths_file', None),
            batch_size=getattr(config, 'inference_batch_size', 4),
            num_steps=config.sampling_steps,
            cache_manager=cache_manager # Pass cache manager
        )

        if is_main_process(rank):
            logger.info(f"Inference finished. Outputs saved to: {output_dir}")

    except Exception as e:
        # Log exception using the logger if it was initialized, otherwise print
        if logger:
            logger.exception(f"An error occurred during inference: {e}")
        else:
            print(f"ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
        # Potentially exit with error code
        # sys.exit(1)

    finally:
        # --- Cleanup ---
        # Wait for logging thread - Check if log_queue was created AND is not None
        # Check if rank is main process and wandb was enabled *during setup*
        # The `log_queue is not None` check prevents NameError
        if is_main_process(rank) and getattr(config, 'wandb_enabled', False) and log_queue is not None:
            logger.info("Waiting for background logging to complete...")
            log_queue.put(None) # Send sentinel
            # Optional: Wait for queue processing if tasks are tracked
            # log_queue.join()
            # Optional: Wait for thread exit (depends on daemon setting and if join is needed)
            # log_thread.join() # log_thread is local to setup_environment, cannot join here directly
                                # Consider returning log_thread from setup_environment if joining is essential

        # Finish WandB run
        if is_main_process(rank) and getattr(config, 'wandb_enabled', False):
            try:
                import wandb
                if wandb.run:
                    wandb.finish()
                    logger.info("WandB run finished.")
            except Exception as e:
                 logger.error(f"Error finishing WandB run: {e}")

        # Cleanup distributed environment
        if is_dist_initialized():
            logger.info(f"Rank {rank}: Destroying process group.")
            dist.destroy_process_group()

if __name__ == "__main__":
    # Consider adding argparse here for command-line overrides
    main() 