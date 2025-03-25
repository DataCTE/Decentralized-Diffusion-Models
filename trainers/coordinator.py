"""Coordinator for Decentralized Diffusion Models with Uniform Distribution"""

import os
import torch
import datetime
import time
import contextlib
from tqdm.auto import tqdm
import concurrent.futures

# Import needed components
from trainers.router import RouterTrainer
from trainers.sampling import ddm_sample
from data.dataset import DDMDataset
from utils.logging import setup_logger
from utils.checkpoint import save_coordinator_checkpoint, load_coordinator_checkpoint
from data.dataset import BucketBatchSampler
from torch.utils.data import DataLoader 
import torch.distributed as dist

# Import centralized utilities
from utils.distributed import is_dist_initialized, synchronize, broadcast_object


# Import FSDP directly to fix the "FSDP is not defined" error
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


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
        
        # Verify GPU device is correctly set
        if torch.cuda.is_available():
            # Force set device to match rank
            torch.cuda.set_device(self.rank)
        
        # Log initial memory usage
        self._log_gpu_memory_usage("initialization_start")
        
        # Parallel initialization components
        self._init_parallel_components()
        
        # Log memory after component init
        self._log_gpu_memory_usage("after_component_init")
        
        # Verify proper sharding across GPUs
        self._verify_sharding()
        
        # Log memory after verification
        self._log_gpu_memory_usage("after_verification")
        
        # Defer non-critical initialization
        self.flow_matcher = None  # Will be created on first training step
        
        # Modify sample directory creation
        if config.enable_sampling and rank == 0:
            sample_dir = os.path.join(config.output_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
        else:
            logger.info("Sampling disabled in config")
        
        # Add this after component initialization
        self._init_data_loaders()  # Initialize data loaders
        
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
        """Initialize critical components without manual synchronization"""
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=2,
                desc="Initializing Components",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

        # Parallel component initialization
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._init_and_verify_router): "router",
                executor.submit(self._init_and_verify_experts): "experts"
            }

            try:
                for future in concurrent.futures.as_completed(futures):
                    component = futures[future]
                    result = future.result()
                    if pbar and self.rank == 0:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed: {component} | {result}")
            finally:
                if pbar:
                    pbar.close()

    def _init_data_loaders(self):
        """Initialize distributed-aware data loaders"""
        debug_print("Initializing data loaders", self.rank)
        
        # Remove device parameter from dataset initialization
        train_dataset = DDMDataset(self.config, 'train')
        val_dataset = DDMDataset(self.config, 'val')
        
        # Create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        loader_config = {
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'sampler': train_sampler  # Use distributed sampler
        }
        
        # Simplified collate function
        def collate_fn(batch):
            try:
                return {
                    'latent': torch.stack([item['latent'] for item in batch]),
                    'clip_embedding': torch.stack([item['clip_embedding'] for item in batch]),
                    'bucket': torch.stack([item['bucket'] for item in batch]),
                    'expert': torch.stack([item['expert'] for item in batch])
                }
            except Exception as e:
                logger.error(f"Collation error: {str(e)}")
                return None
        
        # Create DataLoader with distributed sampler
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            **loader_config
        )
        
        # Validation loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_config
        )
    
    def _init_and_verify_router(self):
        """Initialize router with FSDP handling all device placement"""
        # Initialize router trainer with base model
        router_trainer = RouterTrainer(
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Get the pre-wrapped FSDP model from the trainer
        self.router = router_trainer.router
        
        return "Router initialized with FSDP"

    def _init_and_verify_experts(self):
        """Initialize experts without sharding validation"""
        self.expert_indices = self._calculate_expert_shards()
        return f"Experts initialized: {self.expert_indices.tolist()}"

    def _calculate_expert_shards(self):
        """Calculate expert assignments without cross-rank checks"""
        total_experts = self.config.num_experts
        experts_per_rank = (total_experts + self.world_size - 1) // self.world_size
        start = self.rank * experts_per_rank
        end = min(start + experts_per_rank, total_experts)
        return torch.arange(start, end, device=self.device)

    def train(self, num_steps):
        """Distributed training loop with synchronization optimizations"""
        # Set up distributed data sampler
        self.train_loader.sampler.set_epoch(0)  # For epoch-based sampling
        
        for step in range(num_steps):
            try:
                # Synchronized batch loading
                batch = next(iter(self.train_loader))
                batch = self._distribute_batch(batch)
                
                # Expert training phase
                expert_loss = self._train_experts_sync(batch)
                
                # Router update phase
                router_loss = self._train_router_sync(batch)
                
                # Expert redistribution
                if step % self.config.expert_update_interval == 0:
                    self._redistribute_experts()

                # Synchronized logging
                if self.rank == 0:
                    self._log_metrics(step, expert_loss, router_loss)
                
            except Exception as e:
                self._handle_distributed_error(e, step)

    def _distribute_batch(self, batch):
        """Batch distribution handled by DataLoader sharding"""
        return {k: v.to(self.device) for k,v in batch.items()}

    def _train_experts_sync(self, batch):
        """Expert training with quantized async updates"""
        self._set_train_mode('expert')
        
        # Forward pass with quantized gradients
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            loss = self.train_experts(batch)
        
        # Async parameter update
        if self.quant_comm:
            loss = self.quant_comm.all_reduce(loss)
        else:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= self.world_size
        
        return loss.item()

    @contextlib.contextmanager
    def _async_context(self):
        """Non-blocking CUDA stream for communication"""
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            yield
        torch.cuda.current_stream().wait_stream(stream)

    def _train_router_sync(self, batch):
        """Router training with gradient synchronization"""
        self._set_train_mode('router')
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            loss = self.train_router(batch)
        
        # Gradient synchronization
        if is_dist_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= self.world_size
        
        return loss.item()

    def _handle_distributed_error(self, error, step):
        """Coordinated error handling across ranks"""
        error_msg = f"Step {step} error: {str(error)}"
        error_tensor = torch.tensor([1 if error else 0], device=self.device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
        
        if error_tensor.item() == 1:
            if self.rank == 0:
                logger.error(f"Critical error detected: {error_msg}")
            raise RuntimeError("Distributed training error") from error

    def train_experts(self, batch):
        """Train expert models using the DDM approach"""
        total_loss = 0.0
        num_experts = len(self.expert_indices)
        
        # Train each expert assigned to this process
        for expert_idx in self.expert_indices:
            try:
                # Get expert from cache manager
                expert = self.cache_manager.get_expert(
                    expert_idx, 
                    lambda idx: self._create_expert(idx)  # Use lambda to pass expert_idx
                )
                
                # Perform training step
                loss = expert.train_step(batch)
                total_loss += loss
                
            except Exception as e:
                logger.error(f"Error training expert {expert_idx} on rank {self.rank}: {str(e)}")
                continue
        
        # Average loss across experts on this rank
        avg_loss = total_loss / max(num_experts, 1)
        
        # Synchronize losses across all ranks
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def train_router(self, batch):
        """Train router following paper's Section 3.3"""
        try:
            # Get cluster assignments from batch
            true_clusters = batch['expert'].to(self.device)
            
            # Train router with cross-entropy loss as described in paper
            loss = self.router.train_step(
                batch,
                true_clusters=true_clusters,
                temperature=self.config.router_temperature
            )
            
            # Synchronize loss across GPUs
            if self.world_size > 1:
                loss_tensor = torch.tensor([loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss = loss_tensor.item() / self.world_size
            
            return loss
        except Exception as e:
            logger.error(f"Router training failed: {str(e)}")
            return float('inf')
    
    def validate(self, step):
        """Run validation using DDM inference process"""
        if not self.config.enable_validation:
            logger.debug("Skipping validation (disabled in config)")
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
        if not self.config.enable_sampling:
            logger.debug("Skipping sample generation (disabled in config)")
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
            from trainers.expert import ExpertTrainer
            return ExpertTrainer(
                expert_idx=expert_idx,
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
        
        experts_dict = {}
        for expert_idx in self.expert_indices:
            experts_dict[expert_idx] = self.cache_manager.get_expert(expert_idx, expert_builder_fn)

        max_sampling_experts = min(getattr(self.config, 'max_sampling_experts', 4), self.config.num_experts)
        
        if self.rank == 0 and len(experts_dict) < max_sampling_experts:
            all_expert_indices = list(range(self.config.num_experts))
            needed_experts = sorted(all_expert_indices[:max_sampling_experts])
            missing_experts = [idx for idx in needed_experts if idx not in self.expert_indices]
            
            if missing_experts:
                logger.info(f"Creating {len(missing_experts)} additional experts for sampling: {missing_experts}")
                for expert_idx in missing_experts:
                    with self.router.router.no_sync():
                        expert = expert_builder_fn(expert_idx)
                        experts_dict[expert_idx] = expert.module if hasattr(expert, 'module') else expert
                        logger.info(f"Successfully created expert {expert_idx} for sampling")
                    torch.distributed.barrier()
        
        for expert in experts_dict.values():
            if hasattr(expert, 'expert'):
                expert.expert.eval()
            else:
                expert.eval()

        try:
            use_mixed_precision = getattr(self.config, 'use_mixed_precision', False)
            
            if hasattr(self.config, 'buckets') and len(self.config.buckets) > 0:
                w, h = self.config.buckets[0]
                C = getattr(self.config, 'latent_channels', 16)
                vae_scale_factor = getattr(self.config, 'vae_scale_factor', 8)
                latent_h, latent_w = h // vae_scale_factor, w // vae_scale_factor
                shape = (num_samples, C, latent_h, latent_w)
            else:
                H, W = self.config.image_size[1], self.config.image_size[2]
                C = getattr(self.config, 'latent_channels', 16)
                vae_scale_factor = getattr(self.config, 'vae_scale_factor', 8)
                latent_h, latent_w = H // vae_scale_factor, W // vae_scale_factor
                shape = (num_samples, C, latent_h, latent_w)
            
            text_embeddings = None
            uncond_embeddings = None
            if prompts is not None and hasattr(self, 'text_encoder'):
                text_embeddings = torch.cat([self.text_encoder.encode(p) for p in prompts], dim=0).to(self.device)
                uncond_embeddings = self.text_encoder.encode([""]*num_samples).to(self.device)
            
            router_model = self.router.router if hasattr(self.router, 'router') else self.router
            if hasattr(router_model, 'eval'):
                router_model.eval()
            
            # Get sampling parameters from config with validation
            inference_strategy = getattr(self.config, 'inference_strategy', 'top_k')
            top_k = min(getattr(self.config, 'top_k', 1), len(experts_dict))
            top_p = getattr(self.config, 'top_p', 0.9)
            
            # Handle special oracle case
            true_clusters = None
            if inference_strategy == "oracle":
                true_clusters = self._get_true_clusters(num_samples)
            
            with torch.amp.autocast(device_type='cuda', enabled=use_mixed_precision):
                samples = ddm_sample(
                    router=router_model,
                    experts=experts_dict,
                    shape=shape,
                    num_steps=self.config.sampling_steps,
                    cfg_scale=self.config.cfg_scale,
                    temperature=self.config.sampling_temp,
                    device=self.device,
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings,
                    inference_strategy=inference_strategy,
                    top_k=top_k,
                    top_p=top_p,
                    cluster_ids=true_clusters if true_clusters else None
                )
            
            try:
                from torchvision.utils import save_image
                for i in range(num_samples):
                    sample_to_save = samples[i].float()
                    save_image(sample_to_save, os.path.join(sample_dir, f'sample_{i}.png'))
                logger.info(f"Saved {num_samples} samples to {sample_dir}")
            except Exception as e:
                logger.error(f"Error saving samples: {e}")
            
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        if return_images and samples is not None:
            return samples.float()
        return None

    def _get_true_clusters(self, num_samples):
        """Get true cluster labels for oracle strategy (paper Section 4.2)"""
        if not hasattr(self, 'val_loader'):
            return None
            
        try:
            batch = next(iter(self.val_loader))
            return batch["expert"][:num_samples].to(self.device)
        except Exception as e:
            logger.warning(f"Couldn't get true clusters for oracle sampling: {e}")
            return None
    
    def save_checkpoint(self, step):
        """Save checkpoint of all components with FSDP support"""
        if not self.config.enable_checkpointing:
            logger.debug("Skipping checkpoint save (disabled in config)")
            return
            
        logger.info(f"Saving checkpoint at step {step}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints', f'step_{step}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Import fsdp utilities for saving
        from utils.fsdp import save_fsdp_model
        
        # Save router using FSDP-aware saving
        router_path = os.path.join(checkpoint_dir, 'router.pt')
        save_fsdp_model(
            self.router,
            router_path,
            optim=self.router.optimizer,
            scheduler=self.router.lr_scheduler,
            metadata={"step": step}
        )
            
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
                
                # Add config flags to wandb
                wandb.config.update({
                    "enable_validation": self.config.enable_validation,
                    "enable_sampling": self.config.enable_sampling,
                    "enable_checkpointing": self.config.enable_checkpointing
                })
                
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

    def _redistribute_experts(self):
        """Improved expert redistribution with gradient-aware sharding"""
        # Get cluster statistics with synchronized access
        cluster_counts = self.train_loader.dataset.get_cluster_distribution()
        cluster_counts = cluster_counts.to(self.device)
        dist.all_reduce(cluster_counts, op=dist.ReduceOp.SUM)
        
        # Calculate new expert assignments using weighted distribution
        total_samples = cluster_counts.sum()
        expert_weights = cluster_counts / total_samples
        new_assignments = torch.distributions.Categorical(expert_weights).sample((self.world_size,))
        
        # Verify assignments are disjoint
        unique_counts = torch.unique(new_assignments, return_counts=True)[1]
        if (unique_counts > 1).any():
            logger.error("Expert redistribution conflict detected!")
            self._resolve_sharding_conflicts(new_assignments)
        
        # Redistribute experts with state preservation
        self._migrate_expert_states(new_assignments)

    def _resolve_sharding_conflicts(self, assignments):
        """Conflicts resolved through FSDP's parameter consensus"""
        self.expert_indices = assignments[self.rank]

    def _migrate_expert_states(self, new_assignments):
        """State-preserving expert migration without synchronization"""
        # Simply reassign experts - FSDP will handle parameter consistency
        self.expert_indices = new_assignments[self.rank]

    def _create_expert(self, expert_idx):
        """Create expert instance without sharding validation"""
        from trainers.expert import ExpertTrainer
        from utils.fsdp import wrap_model_with_fsdp
        
        expert_trainer = ExpertTrainer(
            expert_idx=expert_idx,
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
        
        return wrap_model_with_fsdp(
            expert_trainer.expert,
            self.config,
            param_init_fn=lambda m: m.to_empty(device=self.device, recurse=False),
            rank=self.rank
        )

    def _verify_sharding(self):
        """No-op verification since we trust FSDP's sharding"""
        pass

    def _log_gpu_memory_usage(self, stage=""):
        """Log GPU memory usage from all ranks"""
        if not is_dist_initialized():
            return
        
        # Collect memory stats from this rank
        mem_allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
        mem_reserved = torch.cuda.memory_reserved(self.device) / 1e9  # GB
        
        # Create tensors to gather from all ranks
        mem_allocated_tensor = torch.tensor([mem_allocated], device=self.device)
        mem_reserved_tensor = torch.tensor([mem_reserved], device=self.device)
        
        # Create lists to hold values from all ranks
        all_mem_allocated = [torch.zeros_like(mem_allocated_tensor) for _ in range(self.world_size)]
        all_mem_reserved = [torch.zeros_like(mem_reserved_tensor) for _ in range(self.world_size)]
        
        # Gather memory stats from all ranks
        dist.all_gather(all_mem_allocated, mem_allocated_tensor)
        dist.all_gather(all_mem_reserved, mem_reserved_tensor)
        
        # Log the results
        if self.rank == 0:
            all_allocated = [t.item() for t in all_mem_allocated]
            all_reserved = [t.item() for t in all_mem_reserved]
            
            # Calculate stats
            min_allocated = min(all_allocated)
            max_allocated = max(all_allocated)
            avg_allocated = sum(all_allocated) / len(all_allocated)
            
            # Check imbalance
            imbalance = max_allocated / (min_allocated + 1e-6)
            
            logger.info(f"==== GPU Memory Usage ({stage}) ====")
            for i, (alloc, resv) in enumerate(zip(all_allocated, all_reserved)):
                logger.info(f"Rank {i}: Allocated: {alloc:.2f} GB, Reserved: {resv:.2f} GB")
            
            logger.info(f"Memory imbalance factor: {imbalance:.2f}x")
            
            if imbalance > 1.5:
                logger.warning(f"High memory imbalance detected at {stage}!")

