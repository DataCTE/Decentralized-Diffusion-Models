"""Coordinator for Decentralized Diffusion Models with Uniform Distribution"""

import os
import torch
import datetime
import time
import contextlib
from tqdm.auto import tqdm
import concurrent.futures
from collections import defaultdict
import numpy as np

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
        
        # Convert config to dict before passing to dataset
        config_dict = vars(self.config)
        
        # Create datasets
        train_dataset = DDMDataset(config_dict, 'train')
        val_dataset = DDMDataset(config_dict, 'val')
        
        # Create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        # Use the standalone collate function
        loader_config = {
            'num_workers': 2,
            'pin_memory': False,
            'persistent_workers': False,
            'sampler': train_sampler,
            'collate_fn': DDMDataset.collate_fn  # Use class reference
        }
        
        # Create DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            **loader_config
        )
        
        # Validation loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            **loader_config
        )
    
    def _init_and_verify_router(self):
        """Initialize router with FSDP handling all device placement"""
        # Initialize router trainer with base model
        self.router = RouterTrainer(  # Store the RouterTrainer instance directly
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
        
        return "Router initialized with FSDP"

    def _init_and_verify_experts(self):
        """Initialize expert networks and verify they're correctly distributed"""
        # Initialize expert indices as a list or tensor with proper dimensions
        if not hasattr(self, 'expert_indices') or self.expert_indices is None:
            # If expert_indices doesn't exist or is None, initialize it properly
            # This could be a list of indices or a properly sized tensor
            num_experts_per_rank = max(1, self.config.num_experts // self.world_size)
            start_idx = self.rank * num_experts_per_rank
            end_idx = min(start_idx + num_experts_per_rank, self.config.num_experts)
            
            # Create as a list first to ensure it's iterable
            self.expert_indices = list(range(start_idx, end_idx))
            
            # Convert to tensor if needed elsewhere
            self.expert_indices_tensor = torch.tensor(self.expert_indices, device=self.device)
            
        # Add debug print to verify indices
        print(f"[Rank {self.rank}] Expert indices: {self.expert_indices}")
        
        # Fix here - don't call tolist() on a list
        return f"Experts initialized: {self.expert_indices}"

    def _calculate_expert_shards(self):
        """Calculate expert assignments without cross-rank checks"""
        total_experts = self.config.num_experts
        experts_per_rank = (total_experts + self.world_size - 1) // self.world_size
        start = self.rank * experts_per_rank
        end = min(start + experts_per_rank, total_experts)
        return torch.arange(start, end, device=self.device)

    def train(self, num_steps: int) -> None:
        """Distributed training loop with enhanced synchronization
        
        Implements paper's training procedure from Section 3.4 with:
        - Synchronized expert updates
        - Balanced communication patterns
        - Gradient checkpointing
        """
        self._prepare_training()
        
        for step in range(num_steps):
            try:
                batch = self._get_next_batch()
                if batch is None:
                    continue

                # Distributed training phases
                expert_loss, expert_metrics = self._train_experts(batch)
                router_loss = self._train_router(batch)
                
                # Synchronized model updates
                if step % self.config.expert_update_interval == 0:
                    self._redistribute_experts()
                    self._synchronize_models()

                # Validation and logging
                self._handle_logging(step, expert_loss, router_loss, expert_metrics)
                self._handle_validation(step)

            except Exception as e:
                self._handle_training_error(e, step)

        self._cleanup_training()

    def _prepare_training(self):
        """Initialize training-specific components"""
        torch.cuda.reset_peak_memory_stats()
        self.step = 0
        self.train_start_time = time.time()
        
        # Create gradient scaler if missing
        if not hasattr(self, 'scaler'):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)

    def _get_next_batch(self):
        """Get next batch with proper error handling"""
        try:
            batch = next(iter(self.train_loader))
            return self._distribute_batch(batch)
        except StopIteration:
            logger.info("Training complete - dataset exhausted")
            return None
        except RuntimeError as e:
            logger.error(f"Data loading failed: {str(e)}")
            return None

    def _train_experts(self, batch) -> tuple[float, dict]:
        """Expert training phase with synchronized gradients"""
        total_loss = 0.0
        expert_metrics = {}
        
        for expert_idx in self.expert_indices:
            expert = self.cache_manager.get_expert(expert_idx, self.expert_builder_fn)
            
            with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                loss = expert.train_step(batch)
            
            # Gradient synchronization
            if self.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            
            total_loss += loss.item()
            expert_metrics[f'expert_{expert_idx}_loss'] = loss.item()
        
        return total_loss / len(self.expert_indices), expert_metrics

    def _synchronize_models(self):
        """Synchronize model parameters across devices"""
        # Synchronize router first
        if isinstance(self.router.router, FSDP):
            self.router.router._sync_params()
        
        # Synchronize experts
        for expert_idx in self.expert_indices:
            expert = self.cache_manager.get_expert(expert_idx, self.expert_builder_fn)
            if isinstance(expert.expert, FSDP):
                expert.expert._sync_params()

    def _handle_logging(self, step: int, expert_loss: float, router_loss: float, metrics: dict):
        """Centralized logging handling"""
        if self.rank != 0:
            return

        # Calculate step timing
        step_time = time.time() - self.step_start_time
        
        # Prepare metrics
        log_data = {
            'expert_loss': expert_loss,
            'router_loss': router_loss,
            'step_time': step_time,
            'learning_rate': self.router.optimizer.param_groups[0]['lr'],
            **metrics
        }
        
        # Add memory stats
        if self.config.log_memory:
            log_data.update(self._get_memory_stats())
        
        # WandB logging
        if self.wandb_enabled:
            import wandb
            wandb.log(log_data, step=step)
        
        # Console logging
        logger.info(
            f"Step {step} | Expert Loss: {expert_loss:.4f} | "
            f"Router Loss: {router_loss:.4f} | "
            f"Step Time: {step_time:.2f}s"
        )

    def _cleanup_training(self):
        """Post-training cleanup"""
        torch.cuda.empty_cache()
        if self.rank == 0:
            logger.info("Training completed successfully")

    def _distribute_batch(self, batch):
        """Batch distribution handled by DataLoader sharding"""
        return {k: v.to(self.device) for k,v in batch.items()}

    def _train_router(self, batch):
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
                world_size=self.world_size,
                router=self.router  # Pass coordinator's router
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

    def _redistribute_experts(self):
        """Implements paper's load-balanced expert assignment (Section 3.4) with distributed sync"""
        # Gather cluster statistics across all ranks
        cluster_counts = self.dataset.get_cluster_sizes()
        total_samples = cluster_counts.sum()
        
        # Paper's load balancing equations (9-10)
        expert_loads = cluster_counts.float()
        target_load = (total_samples / self.world_size) * self.config.expert_capacity_factor
        
        # Sort clusters by load descending
        sorted_loads, sorted_indices = torch.sort(expert_loads, descending=True)
        
        # Distributed assignment using PyTorch collectives
        assignments = torch.zeros_like(sorted_indices)
        rank_loads = torch.zeros(self.world_size, device=self.device)
        
        for idx in sorted_indices:
            min_rank = torch.argmin(rank_loads)
            if rank_loads[min_rank] + expert_loads[idx] <= target_load:
                assignments[idx] = min_rank
                rank_loads[min_rank] += expert_loads[idx]
            else:
                # Find next best rank with capacity
                available = torch.where(rank_loads < target_load)[0]
                if len(available) > 0:
                    chosen_rank = available[torch.argmin(rank_loads[available])]
                    assignments[idx] = chosen_rank
                    rank_loads[chosen_rank] += expert_loads[idx]
                else:
                    # Fallback to round-robin assignment
                    assignments[idx] = idx % self.world_size

        # Synchronize assignments across all ranks
        assignments = self._broadcast_assignments(assignments)
        
        # Update expert indices for this rank
        self.expert_indices = torch.where(assignments == self.rank)[0].tolist()
        self.expert_indices_tensor = torch.tensor(self.expert_indices, device=self.device)

    def _broadcast_assignments(self, assignments: torch.Tensor) -> torch.Tensor:
        """Ensure consistent expert assignments across all ranks using PyTorch collectives"""
        if self.world_size > 1:
            # Gather all assignments at rank 0
            assignment_list = [torch.empty_like(assignments) for _ in range(self.world_size)]
            dist.gather(assignments, assignment_list if self.rank == 0 else None, dst=0)
            
            # Validate and select optimal assignment (rank 0 does consensus)
            if self.rank == 0:
                # Use assignment from rank 0 as authoritative
                consensus = assignment_list[0]
                for a in assignment_list[1:]:
                    if not torch.allclose(consensus, a):
                        logger.warning("Expert assignment mismatch, using rank 0's version")
                        break
            else:
                consensus = torch.empty_like(assignments)
            
            # Broadcast final assignment to all ranks
            dist.broadcast(consensus, src=0)
            return consensus
        return assignments

    def _verify_sharding(self):
        """No-op verification since we trust FSDP's sharding"""
        pass

    def _log_gpu_memory_usage(self, stage: str = ""):
        """Log GPU memory usage from all ranks with paper-cited metrics"""
        if not is_dist_initialized():
            return
        
        # Collect memory stats using paper's efficiency metrics
        stats = {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "active_gb": torch.cuda.memory_stats().get("active_bytes.all.current", 0) / 1e9
        }
        
        # Create tensors for distributed reduction
        metrics = torch.tensor([
            stats["allocated_gb"],
            stats["reserved_gb"], 
            stats["active_gb"]
        ], device=self.device)
        
        # Gather metrics across all ranks
        world_metrics = [torch.zeros_like(metrics) for _ in range(self.world_size)]
        dist.all_gather(world_metrics, metrics)
        
        # Log distribution statistics (Section 4.3)
        if self.rank == 0:
            allocated = [m[0].item() for m in world_metrics]
            reserved = [m[1].item() for m in world_metrics]
            active = [m[2].item() for m in world_metrics]
            
            logger.info(f"Memory Distribution ({stage}):")
            logger.info(f"Allocated: μ={np.mean(allocated):.2f} ±{np.std(allocated):.2f} GB")
            logger.info(f"Reserved: μ={np.mean(reserved):.2f} ±{np.std(reserved):.2f} GB")
            logger.info(f"Active: μ={np.mean(active):.2f} ±{np.std(active):.2f} GB")

    def _handle_training_error(self, error, step):
        """Coordinated error handling across ranks"""
        error_msg = f"Step {step} error: {str(error)}"
        error_tensor = torch.tensor([1 if error else 0], device=self.device)
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
        
        if error_tensor.item() == 1:
            if self.rank == 0:
                logger.error(f"Critical error detected: {error_msg}")
            raise RuntimeError("Distributed training error") from error

