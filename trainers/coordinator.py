"""Coordinator for Decentralized Diffusion Models with Uniform Distribution"""

import os
import torch
import datetime
import time
from collections import defaultdict

from tqdm.auto import tqdm
import concurrent.futures

# Import needed components
from trainers.router import RouterTrainer
from trainers.sampling import ddm_sample
from trainers.diffusion import DecentralizedFlowMatcher
from data.dataset import DDMDataset
from utils.logging import setup_logger
from utils.checkpoint import save_coordinator_checkpoint, load_coordinator_checkpoint
from data.dataset import BucketBatchSampler
from torch.utils.data import DataLoader 



# Setup logger
logger = setup_logger("DDMCoordinator")

# Direct console print function for immediate feedback
def debug_print(message, rank=None, force=False):
    """Print directly to console regardless of logger configuration"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if rank is not None:
        prefix = f"[DDMT-{rank}]"
    else:
        prefix = "[DDMT]"
        
    if force or (rank is not None and rank == 0) or rank is None:
        print(f"{prefix} [{timestamp}] {message}", flush=True)

class DDMTrainingCoordinator:
    """Coordinator for Decentralized Diffusion Models with uniform data distribution"""
    
    def __init__(self, config, rank, world_size, cache_manager=None, progress_callback=None):
        """
        Initialize coordinator for decentralized diffusion
        
        Args:
            config: Configuration object
            rank: Process rank (0 is main)
            world_size: Total number of processes
            cache_manager: Optional cache manager
            progress_callback: Optional callback function to report initialization progress
        """
        init_start_time = time.time()
        debug_print(f"Starting DDM initialization on rank {rank}/{world_size}", rank, force=True)
        
        # Store basic configuration
        self.config = config
        
        # Add verbose flag with default value
        self.verbose = getattr(config, 'verbose_training', False)
        
        # Ensure all required configuration parameters exist
        self._ensure_config_completeness()
        
        self.rank = rank
        self.world_size = world_size
        self.progress_callback = progress_callback
        self.cache_manager = cache_manager
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        # Initialize wandb - only on rank 0
        self._init_wandb()
        
        # Parallel initialization components
        self._init_parallel_components()
        
        # Defer non-critical initialization
        self.flow_matcher = None  # Will be created on first training step
        
        # Final initialization sync
        total_init_time = time.time() - init_start_time
        debug_print(f"DDM initialization completed in {total_init_time:.2f}s", rank, force=True)
        
        # Log initialization info to wandb
        if self.rank == 0 and self.wandb_enabled:
            import wandb
            wandb.log({"initialization_time": total_init_time})
    
    def _ensure_config_completeness(self):
        """
        Ensure that the config has all required parameters.
        This is a minimal check, since most defaults are handled in config.py.
        """
        # Map hidden_size to hidden_dim if hidden_dim isn't present but hidden_size is
        if not hasattr(self.config, 'hidden_dim') and hasattr(self.config, 'hidden_size'):
            self.config.hidden_dim = self.config.hidden_size
            logger.info(f"Mapped config.hidden_size to config.hidden_dim = {self.config.hidden_dim}")
        
        # Set ffn_dim if not present but can be derived from hidden_dim
        if not hasattr(self.config, 'ffn_dim') and hasattr(self.config, 'hidden_dim'):
            self.config.ffn_dim = self.config.hidden_dim * 4
            logger.info(f"Derived config.ffn_dim from hidden_dim = {self.config.ffn_dim}")
        
        # Log a note about using the 16ch-VAE model
        if hasattr(self.config, 'vae_model') and "16ch-vae" in self.config.vae_model:
            if getattr(self.config, 'latent_channels', 0) != 16:
                self.config.latent_channels = 16
                logger.warning(f"Enforced latent_channels=16 for 16ch-VAE compatibility")
    
    def _init_parallel_components(self):
        """Initialize critical components with async dataset loading"""
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=2,  # Reduced from 3 to 2 (router and experts only)
                desc="Initializing Components",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

        # Use ThreadPoolExecutor for better resource management
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit data loading separately without tracking
            data_future = executor.submit(self._init_data_loaders)
            
            # Only track these two components in progress bar
            futures = {
                executor.submit(self._init_router): "router",
                executor.submit(self._init_expert_indices): "experts"
            }

            try:
                # Process completion with progress updates
                for future in concurrent.futures.as_completed(futures):
                    component = futures[future]
                    future.result()  # Raise exceptions if any
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed: {component}")
            finally:
                if pbar is not None:
                    pbar.close()
            
            # Ensure data loading completes before continuing
            data_future.result()
    
    def _init_data_loaders(self):
        """Initialize data loaders without multiprocessing to avoid pickling requirements"""
        debug_print(f"Initializing data loaders on rank {self.rank}", self.rank)
        
        # Shared configuration for DataLoader
        loader_config = {
            'num_workers': 0,  # Use single-process data loading
            'pin_memory': False,  # This is safe to use without multiprocessing
        }
        
        # Train dataset
        train_dataset = DDMDataset(self.config, 'train')
        
        # Create bucket indices directly from bucket_assignments
        if hasattr(train_dataset, 'bucket_assignments'):
            bucket_indices = defaultdict(list)
            for idx, bucket_idx in enumerate(train_dataset.bucket_assignments.cpu().numpy()):
                bucket_indices[int(bucket_idx)].append(idx)
            
            # Use our existing BucketBatchSampler
            if bucket_indices:
                debug_print(f"Creating BucketBatchSampler with batch size {self.config.batch_size}", self.rank)
                batch_sampler = BucketBatchSampler(
                    bucket_indices=bucket_indices,
                    batch_size=self.config.batch_size,
                    device='cpu',  # Use CPU for initial setup
                    shuffle=True,
                    drop_last=False
                )
                
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=batch_sampler,
                    **loader_config
                )
                if self.rank == 0:
                    logger.info(f"Created bucket-aware DataLoader with {len(bucket_indices)} buckets and batch size {self.config.batch_size}")
            else:
                # Fallback to simple loader with batch_size=1
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=True,
                    **loader_config
                )
        else:
            # Fallback to simple loader if bucket_assignments isn't available
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                **loader_config
            )
        
        # Validation dataset - always use batch_size=1 for safety
        val_dataset = DDMDataset(self.config, 'val')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            **loader_config
        )
    
    def _init_expert_indices(self):
        """Determine expert assignments without model creation"""
        self.expert_indices = [
            idx for idx in range(self.config.num_experts)
            if idx % self.world_size == self.rank
        ]
        logger.info(f"Rank {self.rank} will manage {len(self.expert_indices)} experts")
    
    def _init_router(self):
        """Initialize router with async FSDP wrapping"""
        logger.info(f"Creating router on rank {self.rank}")
        self.router = RouterTrainer(
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
    
    def train(self, num_steps):
        """Train the DDM system for the specified number of steps"""
        if num_steps <= 0:
            logger.warning(f"Invalid step count {num_steps}, skipping training")
            return
            
        logger.info(f"Starting DDM training for {num_steps} steps on rank {self.rank}")
        
        # Initialize flow matcher on first use
        if self.flow_matcher is None:
            self.flow_matcher = DecentralizedFlowMatcher(
                sigma=getattr(self.config, 'sigma', 0.5),
                loss_type=getattr(self.config, 'loss_type', 'huber')
            )
        
        # Watch models in wandb if requested and not done yet
        if self.rank == 0 and self.wandb_enabled and hasattr(self, 'wandb_watch_model') and self.wandb_watch_model:
            import wandb
            # Watch the router model
            wandb.watch(
                self.router.router, 
                log=self.wandb_watch_model,
                log_freq=getattr(self.config, 'wandb_log_every', 1)
            )
            # We'd need to watch expert models too, but they're managed by cache_manager
        
        # Train loop implementing the DDM training approach
        global_step = 0  # Initialize global step counter
        start_time = time.time()
        
        for step in range(num_steps):
            # Get a batch by creating a fresh iterator each time to avoid pickling issues
            try:
                batch = next(iter(self.train_loader))
            except Exception as e:
                logger.error(f"Error getting batch at step {step}: {str(e)}")
                continue
            
            step_start_time = time.time()
            
            # Joint training of experts and router
            expert_loss = self.train_experts(batch)  # Updates experts
            router_loss = self.train_router(batch)   # Updates router
            
            step_duration = time.time() - step_start_time
            global_step += 1
            
            # Log metrics to wandb on rank 0
            if self.rank == 0 and self.wandb_enabled:
                self._log_step_metrics_to_wandb(
                    step=global_step,
                    expert_loss=expert_loss,
                    router_loss=router_loss,
                    step_duration=step_duration,
                    learning_rates=self._get_learning_rates(),
                    memory_stats=self._get_memory_stats() if getattr(self.config, 'wandb_log_memory', True) else None
                )
            
            # Log every N steps
            if step % 100 == 0 or step == num_steps - 1:
                logger.info(f"Step {step}/{num_steps}: Expert loss = {expert_loss:.4f}, Router loss = {router_loss:.4f}")
                
            # Periodic validation
            if step % 1000 == 0:
                self.validate(step)
                
            # Save checkpoint every N steps
            if step > 0 and step % 5000 == 0:
                self.save_checkpoint(step)
        
        # Log final training stats
        total_duration = time.time() - start_time
        if self.rank == 0 and self.wandb_enabled:
            import wandb
            import threading
            
            # Define a thread for final logging
            def final_log_thread():
                try:
                    wandb.log({
                        "training_complete": True,
                        "total_training_duration": total_duration,
                        "steps_per_second": num_steps / total_duration
                    }, commit=True)  # Use commit=True for final log
                except Exception as e:
                    print(f"Warning: W&B final logging error: {e}")
            
            # Start logging in a separate thread
            threading.Thread(target=final_log_thread).start()
            
            # Flush any remaining logs
            self.flush_wandb_logs()
    
    def train_experts(self, batch):
        """Train expert models using the DDM approach"""
        total_loss = 0.0
        
        # Define the expert builder function that will be used to create experts when needed
        def expert_builder_fn(expert_idx):
            # Import the actual expert trainer class we have
            from trainers.expert import ExpertTrainer
            
            # Create a new expert trainer with proper initialization 
            expert = ExpertTrainer(
                expert_idx=expert_idx,
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
            return expert
        
        # Train each expert assigned to this process
        for expert_idx in self.expert_indices:
            # Get expert from cache manager with the builder function
            expert = self.cache_manager.get_expert(expert_idx, expert_builder_fn)
            
            # Perform training step and accumulate loss
            loss = expert.train_step(batch)
            total_loss += loss
        
        # Return average loss across experts
        return total_loss / max(len(self.expert_indices), 1)
    
    def train_router(self, batch):
        """Train the router with the provided batch"""
        # Remove reference to non-existent self.verbose attribute
        # Replace with a config-based check or default to False
        verbose = getattr(self.config, 'verbose_router_training', False)
        
        if self.rank == 0 and verbose:
            logger.debug("Training router...")
        
        try:
            # Train step now doesn't require cluster_idx
            loss = self.router.train_step(batch)
            return loss
        except Exception as e:
            logger.error(f"Router training failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')  # Return a placeholder value to continue training
    
    def validate(self, step):
        """Run validation using DDM inference process"""
        # Only run validation on rank 0
        if self.rank != 0:
            return
            
        logger.info(f"Running validation at step {step}")
        
        # Generate samples using DDM sampling
        sample_images = self.generate_samples(num_samples=4, step=step, return_images=True)
        
        # Log samples to wandb
        if self.wandb_enabled and sample_images:
            import wandb
            # Log images
            wandb.log({
                "validation/samples": [wandb.Image(img) for img in sample_images],
                "validation/step": step
            })
    
    def generate_samples(self, num_samples=4, step=None, prompts=None, return_images=False):
        """Generate samples using the DDM inference approach"""
        if self.rank != 0:
            return None
            
        logger.info(f"Generating {num_samples} samples")
        
        # Initialize samples to None in case of errors
        samples = None
        
        # Create sample directory
        if step is not None:
            sample_dir = os.path.join(self.config.output_dir, 'samples', f'step_{step}')
        else:
            sample_dir = os.path.join(self.config.output_dir, 'samples')
            
        os.makedirs(sample_dir, exist_ok=True)
        
        # Define expert builder function for cache manager
        def expert_builder_fn(expert_idx):
            # Import the actual expert trainer class 
            from trainers.expert import ExpertTrainer
            
            # Create a new expert trainer with proper initialization 
            expert = ExpertTrainer(
                expert_idx=expert_idx,
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
            return expert
        
        # Implement expert sharing - Get access to more experts
        # Approach: For rank 0, create additional experts beyond its assigned ones
        experts_dict = {}
        
        # First collect local experts
        for expert_idx in self.expert_indices:
            experts_dict[expert_idx] = self.cache_manager.get_expert(expert_idx, expert_builder_fn)
        
        # For rank 0, also create additional experts for better sampling
        # Define max number of experts to use for sampling
        max_sampling_experts = min(getattr(self.config, 'max_sampling_experts', 4), self.config.num_experts)
        
        # Add additional experts if needed (up to max_sampling_experts)
        if self.rank == 0 and len(experts_dict) < max_sampling_experts:
            # Get indices of experts we need to create
            all_expert_indices = list(range(self.config.num_experts))
            needed_experts = sorted(all_expert_indices[:max_sampling_experts])
            missing_experts = [idx for idx in needed_experts if idx not in self.expert_indices]
            
            if missing_experts:
                logger.info(f"Creating {len(missing_experts)} additional experts for sampling: {missing_experts}")
                # Ensure all ranks participate in expert creation
                for expert_idx in missing_experts:
                    # Create expert with proper distributed context
                    with self.router.router.no_sync():  # Disable router gradient sync
                        expert = expert_builder_fn(expert_idx)
                        if hasattr(expert, 'module'):  # Unwrap DDP/FSDP
                            experts_dict[expert_idx] = expert.module
                        else:
                            experts_dict[expert_idx] = expert
                        logger.info(f"Successfully created expert {expert_idx} for sampling")
                    # Synchronize after expert creation
                    torch.distributed.barrier()
        
        # Put experts in evaluation mode before sampling
        for expert in experts_dict.values():
            if hasattr(expert, 'expert'):
                expert.expert.eval()  # For ExpertTrainer objects
            else:
                expert.eval()  # For direct model objects
        
        # Validate top_k value
        top_k = getattr(self.config, 'top_k', 1)
        num_available_experts = len(experts_dict)
        if top_k > num_available_experts:
            logger.warning(f"top_k ({top_k}) is greater than available experts ({num_available_experts}). Setting top_k={num_available_experts}")
            top_k = num_available_experts
        
        # Use proper DDM sampling from trainers/sampling.py
        try:
            # Determine whether to use mixed precision
            use_mixed_precision = getattr(self.config, 'use_mixed_precision', False)
            
            # Use the first bucket's dimensions for sampling
            # In real applications, you might want to sample from different buckets
            if hasattr(self.config, 'buckets') and len(self.config.buckets) > 0:
                w, h = self.config.buckets[0]  # Get dimensions from first bucket
                
                # Use latent_channels instead of image_size[0]
                C = getattr(self.config, 'latent_channels', 16)  # Default to 16 for 16ch-VAE
                
                # Adjust dimensions for VAE latent space if needed
                vae_scale_factor = getattr(self.config, 'vae_scale_factor', 8)
                latent_h, latent_w = h // vae_scale_factor, w // vae_scale_factor
                
                shape = (num_samples, C, latent_h, latent_w)
                logger.info(f"Generating samples with dimensions {shape} (scaled from {w}x{h}) from bucket 0")
            else:
                # Fallback to image_size but ensure we use latent_channels
                H, W = self.config.image_size[1], self.config.image_size[2]  # Only take H and W
                C = getattr(self.config, 'latent_channels', 16)  # Get channel count from config
                
                # Scale down for latent space
                vae_scale_factor = getattr(self.config, 'vae_scale_factor', 8)
                latent_h, latent_w = H // vae_scale_factor, W // vae_scale_factor
                
                shape = (num_samples, C, latent_h, latent_w)
                logger.info(f"Generating samples with dimensions {shape} from image_size")
            
            # Get optional text embeddings if conditional
            text_embeddings = None
            uncond_embeddings = None
            if prompts is not None and hasattr(self, 'text_encoder') and self.text_encoder is not None:
                text_embeddings = []
                for prompt in prompts:
                    text_embeddings.append(self.text_encoder.encode(prompt))
                text_embeddings = torch.cat(text_embeddings, dim=0).to(self.device)
                
                # Create unconditional embeddings (empty string) for classifier-free guidance
                uncond_embeddings = self.text_encoder.encode([""] * num_samples).to(self.device)
            
            # Access the actual router model, not the trainer
            router_model = self.router.router if hasattr(self.router, 'router') else self.router
            
            # Ensure router is in evaluation mode
            if hasattr(router_model, 'eval'):
                router_model.eval()
            
            # Use consistent precision throughout sampling
            with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
                # Use ddm_sample from trainers/sampling.py for proper DDM sampling
                samples = ddm_sample(
                    router=router_model,  # Use the actual model, not the trainer
                    experts=experts_dict,
                    shape=shape,
                    steps=getattr(self.config, 'sampling_steps', 50),
                    top_k=top_k,
                    device=self.device,
                    cfg_scale=getattr(self.config, 'cfg_scale', 7.5),
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    eta=getattr(self.config, 'eta', 0.0),
                    scheduler=getattr(self.config, 'beta_schedule', "cosine"),
                    verbose=True,
                    temperature=getattr(self.config, 'temperature', 1.0),
                    config=self.config  # Add this line to pass the config
                )
            
            # Save samples
            try:
                from torchvision.utils import save_image
                for i in range(num_samples):
                    # Convert to appropriate format for saving if needed
                    sample_to_save = samples[i].float() if samples[i].dtype != torch.float32 else samples[i]
                    save_image(sample_to_save, os.path.join(sample_dir, f'sample_{i}.png'))
                    
                logger.info(f"Saved {num_samples} samples to {sample_dir}")
            except Exception as e:
                logger.error(f"Error saving samples: {e}")
            
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Add detailed traceback
        
        # Return images if requested
        if return_images and samples is not None:
            # Ensure we return float32 tensors for consistency
            return samples.float() if hasattr(samples, 'float') else samples
        return None
    
    def save_checkpoint(self, step):
        """Save checkpoint of all components"""
        # Only save from main process unless configured otherwise
        if self.rank != 0 and not getattr(self.config, 'save_from_all_ranks', False):
            return
            
        logger.info(f"Saving checkpoint at step {step}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints', f'step_{step}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save router using its save_checkpoint method
        self.router.save_checkpoint(checkpoint_dir, step)
            
        # Save coordinator state
        save_coordinator_checkpoint(
            checkpoint_dir, 
            {
                "step": step,
                "config": self.config,
            }
        )
                    
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """Load checkpoint of all components"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load coordinator state
        coordinator_state = load_coordinator_checkpoint(checkpoint_dir)
        step = coordinator_state.get("step", 0) if coordinator_state else 0
        
        # Load router using its load_checkpoint method
        if self.router is not None:
            router_path = os.path.join(checkpoint_dir, 'router.pt')
            if os.path.exists(router_path):
                self.router.load_checkpoint(router_path)
                    
        return step

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        # Only initialize on rank 0
        self.wandb_enabled = getattr(self.config, 'wandb_enabled', False)
        
        if self.rank == 0 and self.wandb_enabled:
            try:
                import wandb
                
                # Get wandb config parameters with defaults
                project = getattr(self.config, 'wandb_project', 'decentralized-diffusion')
                entity = getattr(self.config, 'wandb_entity', None)
                name = getattr(self.config, 'wandb_name', None)
                run_id = getattr(self.config, 'wandb_id', None)
                tags = getattr(self.config, 'wandb_tags', [])
                group = getattr(self.config, 'wandb_group', None)
                mode = getattr(self.config, 'wandb_mode', 'online')
                dir = getattr(self.config, 'wandb_dir', './wandb')
                save_code = getattr(self.config, 'wandb_save_code', True)
                
                # Convert relevant config attributes to a dict, handling non-serializable types
                config_dict = {}
                for k, v in vars(self.config).items():
                    if not k.startswith('_') and not callable(v):
                        # Handle non-serializable types
                        try:
                            # Test if json serializable
                            import json
                            json.dumps({k: v})
                            config_dict[k] = v
                        except (TypeError, OverflowError):
                            # Convert to string if not serializable
                            config_dict[k] = str(v)
                
                # Initialize wandb
                wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    id=run_id,
                    tags=tags,
                    group=group,
                    dir=dir,
                    config=config_dict,
                    mode=mode,
                    save_code=save_code,
                    resume="allow"
                )
                
                # Watch models if requested
                watch_model = getattr(self.config, 'wandb_watch_model', None)
                if watch_model:
                    # We'll watch models after they're initialized
                    self.wandb_watch_model = watch_model
                else:
                    self.wandb_watch_model = None
                
                # Print the dashboard URL prominently
                entity_str = f"{entity}/" if entity else ""
                dashboard_url = f"https://wandb.ai/{entity_str}{project}/runs/{wandb.run.id}"
                
                print("\n" + "=" * 80)
                print(f"W&B Dashboard: {dashboard_url}")
                print("=" * 80 + "\n")
                
                logger.info(f"W&B initialized: {wandb.run.name} (ID: {wandb.run.id})")
                
            except ImportError:
                logger.warning("wandb package not found. Install with 'pip install wandb'")
                self.wandb_enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {str(e)}")
                self.wandb_enabled = False
                # Print exception traceback for debugging
                import traceback
                logger.warning(traceback.format_exc())
        else:
            self.wandb_enabled = False

    def _log_step_metrics_to_wandb(self, step, expert_loss, router_loss, step_duration, learning_rates=None, memory_stats=None):
        """Log per-step metrics to wandb in a truly non-blocking way"""
        # Import in function scope to avoid import errors if wandb is not installed
        import wandb
        import threading
        
        # Prepare metrics dict
        metrics = {
            "train/expert_loss": expert_loss,
            "train/router_loss": router_loss,
            "train/total_loss": expert_loss + router_loss,
            "train/step_duration": step_duration,
            "train/steps_per_second": 1.0 / max(step_duration, 1e-5),
        }
        
        # Add learning rates if provided
        if learning_rates:
            for name, lr in learning_rates.items():
                metrics[f"train/lr_{name}"] = lr
        
        # Add memory stats if provided
        if memory_stats:
            for name, value in memory_stats.items():
                metrics[f"system/{name}"] = value
        
        # Create a thread for non-blocking logging
        def log_thread():
            try:
                # Log metrics to wandb with commit=False to queue up multiple log calls
                wandb.log(metrics, step=step, commit=False)
            except Exception as e:
                # Don't let logging errors crash training
                print(f"Warning: W&B logging error: {e}")
        
        # Start logging in a separate thread
        threading.Thread(target=log_thread).start()
        
        # Get commit frequency from config (default to 10)
        commit_frequency = getattr(self.config, 'wandb_commit_frequency', 10)

        # Every N steps, commit the logs in another thread
        if commit_frequency > 0 and step % commit_frequency == 0:
            def commit_thread():
                try:
                    wandb.log({}, commit=True)  # Empty log with commit=True to flush queue
                except Exception as e:
                    print(f"Warning: W&B commit error: {e}")
            
            threading.Thread(target=commit_thread).start()

    def _get_learning_rates(self):
        """Get current learning rates from all optimizers"""
        lrs = {}
        
        # Router learning rate
        if hasattr(self.router, 'optimizer') and self.router.optimizer:
            for param_group in self.router.optimizer.param_groups:
                lrs["router"] = param_group['lr']
                break
        
        # Expert learning rates - sample from first expert if available
        if hasattr(self, 'expert_indices') and self.expert_indices:
            # Define expert builder function
            def expert_builder_fn(expert_idx):
                # Import the actual expert trainer class 
                from trainers.expert import ExpertTrainer
                
                # Create a new expert trainer with proper initialization 
                expert = ExpertTrainer(
                    expert_idx=expert_idx,
                    config=self.config,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size
                )
                return expert
            
            for expert_idx in self.expert_indices:
                expert = self.cache_manager.get_expert(expert_idx, expert_builder_fn)
                if hasattr(expert, 'optimizer') and expert.optimizer:
                    for param_group in expert.optimizer.param_groups:
                        lrs[f"expert_{expert_idx}"] = param_group['lr']
                        break
                break  # Only get rate for first expert
        
        return lrs

    def _get_memory_stats(self):
        """Get current GPU memory usage"""
        stats = {}
        try:
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1e9
            stats["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
            stats["gpu_max_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 1e9
        except:
            pass
        return stats

    def cleanup(self):
        """Clean up resources on training completion"""
        if self.rank == 0 and self.wandb_enabled:
            import wandb
            # Finish the wandb run
            wandb.finish()
            logger.info("W&B logging completed and run finalized")

    def flush_wandb_logs(self):
        """Flush any pending W&B logs"""
        if self.rank == 0 and self.wandb_enabled:
            try:
                import wandb
                import threading
                
                def flush_thread():
                    try:
                        wandb.log({}, commit=True)  # Force commit any queued logs
                    except Exception as e:
                        print(f"Warning: W&B flush error: {e}")
                
                threading.Thread(target=flush_thread).start()
            except:
                pass

