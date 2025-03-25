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
        
        # Modify sample directory creation
        if config.enable_sampling and rank == 0:
            sample_dir = os.path.join(config.output_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
        else:
            logger.info("Sampling disabled in config")
        
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
        """Initialize critical components with proper thread safety"""
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=2,
                desc="Initializing Components",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

        # Initialize dataset FIRST in main thread to ensure tqdm safety
        self._init_data_loaders()

        # Then parallelize other components
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._init_router): "router",
                executor.submit(self._init_expert_indices): "experts"
            }

            try:
                for future in concurrent.futures.as_completed(futures):
                    component = futures[future]
                    future.result()
                    if pbar and self.rank == 0:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed: {component}")
            finally:
                if pbar:
                    pbar.close()
    
    def _init_data_loaders(self):
        """Initialize data loaders without multiprocessing to avoid pickling requirements"""
        debug_print(f"Initializing data loaders on rank {self.rank}", self.rank)
        
        # Shared configuration for DataLoader - disable multiprocessing
        loader_config = {
            'num_workers': 0,  # Disable multiprocessing
            'pin_memory': False,  # Keep pin_memory disabled
            'persistent_workers': False  # Disable persistent workers
        }
        
        # Initialize training dataset
        train_dataset = DDMDataset(self.config, 'train')
        
        # Create bucket-aware sampler if bucket assignments are available
        if hasattr(train_dataset, 'bucket_assignments'):
            debug_print(f"Creating BucketBatchSampler with batch size {self.config.batch_size}", self.rank)
            
            batch_sampler = BucketBatchSampler(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                device=self.device,
                shuffle=True,
                drop_last=True
            )
            
            # Improved collate function that handles variable sequence lengths
            def collate_fn(batch):
                try:
                    # Handle empty batch
                    if not batch:
                        return None
                        
                    # Get max sequence length in this batch
                    max_seq_len = max(emb.size(1) for item in batch for emb in item['clip_embedding'])
                    
                    # Prepare lists for each key
                    latents, clip_embeddings, buckets, experts = [], [], [], []
                    
                    for item in batch:
                        # Handle latents (already fixed size)
                        latents.append(item['latent'])
                        
                        # Handle CLIP embeddings (need padding)
                        emb = item['clip_embedding']
                        if emb.size(1) < max_seq_len:
                            # Pad with zeros to max length
                            padding = torch.zeros(
                                emb.size(0), 
                                max_seq_len - emb.size(1), 
                                emb.size(2), 
                                device=emb.device
                            )
                            emb = torch.cat([emb, padding], dim=1)
                        clip_embeddings.append(emb)
                        
                        # Handle scalar values
                        buckets.append(item['bucket'])
                        experts.append(item['expert'])
                    
                    # Stack all tensors
                    return {
                        'latent': torch.stack(latents),
                        'clip_embedding': torch.stack(clip_embeddings),
                        'bucket': torch.stack(buckets),
                        'expert': torch.stack(experts)
                    }
                    
                except Exception as e:
                    logger.error(f"Collation error: {str(e)}")
                    return None
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **loader_config
            )
            if self.rank == 0:
                logger.info(f"Created bucket-aware DataLoader with batch size {self.config.batch_size}")
        else:
            # Fallback to simple loader
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=1,
                shuffle=True,
                **loader_config
            )
        
        # Validation dataset with same collate function
        val_dataset = DDMDataset(self.config, 'val')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn if 'collate_fn' in locals() else None,
            **loader_config
        )
    
    def _init_expert_indices(self):
        """
        Determine expert assignments based on paper's Section 3.2 - each expert 
        trains independently on its assigned data cluster
        """
        # Current implementation: Simple round-robin assignment
        self.expert_indices = [
            idx for idx in range(self.config.num_experts)
            if idx % self.world_size == self.rank
        ]
        
        # Should be modified to:
        cluster_sizes = self.train_loader.dataset.get_cluster_sizes()  # Get size of each cluster
        min_size = self.config.min_cluster_samples  # From paper Section 4.1
        
        # Filter out clusters that are too small
        valid_clusters = [
            idx for idx in range(self.config.num_experts)
            if cluster_sizes[idx] >= min_size and idx % self.world_size == self.rank
        ]
        
        self.expert_indices = valid_clusters
        logger.info(f"Rank {self.rank} managing {len(self.expert_indices)} experts with valid cluster sizes")
    
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
        """Train the DDM system following paper's Section 4"""
        print(f"[Rank {self.rank}] Starting training with {num_steps} steps")
        
        # Paper's training phases and intervals
        expert_update_interval = getattr(self.config, 'expert_update_interval', 1000)
        router_update_interval = getattr(self.config, 'router_update_interval', 100)
        
        for step in range(num_steps):
            try:
                step_start_time = time.time()
                
                batch = next(iter(self.train_loader))
                if batch is None:
                    continue
                
                # 1. Expert Training Phase
                expert_loss = self.train_experts(batch)
                
                # 2. Router Update Phase (Section 3.3)
                # Router only trains on rank 0 and at specified intervals
                router_loss = 0.0
                if self.rank == 0 and step % router_update_interval == 0:
                    router_loss = self.train_router(batch)
                
                # 3. Expert Redistribution Phase (Section 4.1)
                if step % expert_update_interval == 0:
                    self._redistribute_experts()
                
                # Calculate step duration
                step_duration = time.time() - step_start_time
                
                # Log metrics
                if self.rank == 0:
                    self._log_step_metrics_to_wandb(
                        step=step,
                        expert_loss=expert_loss,
                        router_loss=router_loss,
                        step_duration=step_duration,
                        learning_rates=self._get_learning_rates(),
                        memory_stats=self._get_memory_stats()
                    )
                
            except Exception as e:
                print(f"[Rank {self.rank}] Critical error in step {step}:")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Current latent files count: {len(self.train_loader.dataset.latent_files)}")
                print(f"Attempted index: {e.args[1] if len(e.args) > 1 else 'N/A'}")
                raise
        
        # Log final training stats
        if self.rank == 0 and self.wandb_enabled:
            import wandb
            import threading
            
            # Define a thread for final logging
            def final_log_thread():
                try:
                    wandb.log({
                        "training_complete": True
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
        """
        Train router following paper's Section 3.3 - router predicts cluster
        probabilities p(k|x_t, t) using cross-entropy loss
        """
        if self.rank == 0:  # Router trains only on rank 0
            try:
                # Get cluster assignments from batch
                true_clusters = batch['expert']
                
                # Train router with cross-entropy loss as described in paper
                loss = self.router.train_step(
                    batch,
                    true_clusters=true_clusters,
                    temperature=self.config.router_temperature
                )
                return loss
            except Exception as e:
                logger.error(f"Router training failed: {str(e)}")
                return float('inf')
        return 0.0
    
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
        """Save checkpoint of all components"""
        if not self.config.enable_checkpointing:
            logger.debug("Skipping checkpoint save (disabled in config)")
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
        """
        Implement paper's expert redistribution strategy (Section 4.1)
        """
        # Get cluster statistics
        cluster_counts = self.train_loader.dataset.get_cluster_distribution()
        active_clusters = torch.where(cluster_counts >= self.config.min_cluster_samples)[0]
        
        # Identify underutilized experts
        for expert_idx in self.expert_indices:
            if cluster_counts[expert_idx] < self.config.min_cluster_samples:
                # Find largest unassigned cluster
                new_cluster = None
                max_count = 0
                for cluster_idx in active_clusters:
                    if (cluster_counts[cluster_idx] > max_count and 
                        cluster_idx not in self.expert_indices):
                        new_cluster = cluster_idx
                        max_count = cluster_counts[cluster_idx]
                
                if new_cluster is not None:
                    # Reassign expert
                    self.expert_indices[self.expert_indices.index(expert_idx)] = new_cluster
                    # Reset expert parameters as described in paper
                    expert = self.cache_manager.get_expert(expert_idx)
                    expert.reset_parameters()
                    logger.info(f"Expert {expert_idx} reassigned to cluster {new_cluster}")

    def _isolate_gradients(self):
        # Implementation of _isolate_gradients method
        pass

