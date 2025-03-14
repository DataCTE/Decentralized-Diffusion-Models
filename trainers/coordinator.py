"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""

import math
import torch
import os
import datetime
import torch.nn.functional as F

from data.dataset import DDMDataset
from data.clustering import ClusterManager
from trainers.Distillation import DiffusionDistiller
from trainers.expert import ExpertTrainer
from trainers.router import RouterTrainer

# Import centralized utilities
from utils.logging import setup_logger, log_metrics, log_images
from utils.distributed import (
    is_main_process, synchronize
)
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from trainers.sampling import ddm_sample
from utils.visualization import tensor_to_pil

# Setup logger
logger = setup_logger("DDMCoordinator")

class DDMTrainingCoordinator:
    """Implements the core training logic from Section 3 of the paper"""
    
    def __init__(self, config, rank, world_size, cache_manager=None):
        """
        Initialize coordinator as per paper Section 4.1
        Args:
            config: Configuration object
            rank: Process rank (0 is main)
            world_size: Total number of processes
            cache_manager: Optional ExpertCacheManager for efficient expert loading/unloading
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.cache_manager = cache_manager
        
        # Initialize clustering
        self.init_cluster_manager()
        
        # Initialize data loaders
        self.init_data_loaders()
        
        # Initialize models
        self.init_models()
        
        # Set up optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        
        # Initialize router optimizer
        self.optimizers['router'] = torch.optim.AdamW(
            self.router.parameters(),
            lr=config.router_learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # Initialize expert optimizers (one per expert)
        for expert_idx in range(config.num_experts):
            if self.is_expert_owned_by_rank(expert_idx):
                # Only create optimizer for experts owned by this rank
                expert = self.get_expert(expert_idx)
                self.optimizers[f'expert_{expert_idx}'] = torch.optim.AdamW(
                    expert.parameters(),
                    lr=config.learning_rate,
                    betas=config.adam_betas,
                    weight_decay=config.weight_decay
                )
        
        # Create learning rate schedulers (cosine decay)
        warmup_steps = getattr(config, 'warmup_steps', 1000)
        total_steps = config.num_steps
        
        for key, optimizer in self.optimizers.items():
            self.schedulers[key] = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(step / warmup_steps, 1.0) if step < warmup_steps else 
                0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            )
        
        # Initialize flow matcher for loss computation
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma,
            loss_type=config.loss_type
        )
        
        # Get diffusion parameters
        self.alphas, self.alpha_bar, _ = get_alphas_and_betas()
        
        # Track metrics
        self.metrics = {
            'step': 0,
            'expert_loss': {},
            'router_loss': 0.0,
            'learning_rates': {}
        }
        
        # Create AMP grad scaler for mixed precision training
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
        
        # Expert data loaders and iterators
        self.expert_iterators = {}
        
        logger.info(f"DDMTrainingCoordinator initialized on rank {rank}/{world_size}")
        
        # If we're the main process, log the configuration
        if self.rank == 0:
            logger.info(f"Training configuration:")
            for key, value in vars(config).items():
                if not key.startswith('_'):
                    logger.info(f"  {key}: {value}")
                    
        # Synchronize to ensure all processes have completed initialization
        synchronize()
        
    def safe_synchronize(self, timeout_seconds=60, name="operation"):
        """
        Safely synchronize all processes with simplified timeout handling
        
        Args:
            timeout_seconds: Maximum time to wait for synchronization
            name: Name of the operation for logging
        """
        if self.world_size <= 1:
            return True  # No need to synchronize for single-process training
        
        try:
            # Simple barrier with timeout
            # This handles the synchronization in a clean way
            torch.distributed.barrier(timeout=datetime.timedelta(seconds=timeout_seconds))
            self.logger.debug(f"Synchronization for '{name}' completed successfully")
            return True
        except torch.distributed.DistBackendError as e:
            self.logger.error(f"Synchronization timeout after {timeout_seconds}s for '{name}': {str(e)}")
            # Attempt recovery by continuing execution
            return False
        except Exception as e:
            self.logger.error(f"Synchronization error for '{name}': {str(e)}")
            return False

    def sync_experts_state(self):
        """
        Synchronize expert states across all processes with enhanced error handling
        
        Returns:
            bool: Whether synchronization was successful
        """
        if self.world_size <= 1:
            return True
        
        sync_success = True
        try:
            # Log synchronization start
            self.logger.info("Synchronizing expert states across processes")
            
            # Gather expert indices from all processes
            all_expert_indices = []
            for i in range(self.world_size):
                # Get experts assigned to rank i
                indices = [idx for idx in range(self.config.num_experts) 
                          if idx % self.world_size == i]
                all_expert_indices.append(indices)
            
            # Create a list to track successfully synchronized experts
            synced_experts = set()
            failed_experts = set()
            
            # For each expert, synchronize from its owner to all other ranks
            for rank, indices in enumerate(all_expert_indices):
                for expert_idx in indices:
                    # Only the owner broadcasts parameters
                    if self.rank == rank:
                        self.logger.debug(f"Broadcasting expert {expert_idx} from rank {rank}")
                    
                    try:
                        # Try to synchronize this expert
                        success = self.safe_synchronize(
                            timeout_seconds=120, 
                            name=f"expert_{expert_idx}_broadcast"
                        )
                        
                        if success:
                            synced_experts.add(expert_idx)
                        else:
                            failed_experts.add(expert_idx)
                            sync_success = False
                    except Exception as e:
                        self.logger.error(f"Error synchronizing expert {expert_idx}: {str(e)}")
                        failed_experts.add(expert_idx)
                        sync_success = False
            
            # Report on synchronization results
            if failed_experts:
                self.logger.warning(f"Failed to synchronize {len(failed_experts)} experts: {sorted(failed_experts)}")
                self.logger.info(f"Successfully synchronized {len(synced_experts)} experts")
            else:
                self.logger.info(f"Expert state synchronization completed successfully for all {len(synced_experts)} experts")
            
            return sync_success
        except Exception as e:
            self.logger.error(f"Expert state synchronization failed: {str(e)}")
            return False

    def init_cluster_manager(self):
        """Initialize cluster manager for expert assignment"""
        self.logger.info("Initializing cluster manager")
        
        try:
            # Create cluster manager
            self.cluster_manager = ClusterManager(
                local_rank=self.rank,
                dataset_path=self.config.dataset_path,
                config=self.config
            )
            
            # Use safe synchronization
            self.safe_synchronize(timeout_seconds=300, name="cluster_manager_init")
            
            self.logger.info("Cluster manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize cluster manager: {str(e)}")
            raise
            
    def init_data_loaders(self):
        """Initialize data loaders for training"""
        self.logger.info("Initializing data loaders")
        
        try:
            # Create dataset
            self.dataset = DDMDataset(
                root_dir=self.config.dataset_path,
                cluster_assignments=self.cluster_manager.get_cluster_assignments(),
                config=self.config,
                vae=self.vae,
                clip=self.clip
            )
            
            # Create expert data loaders
            from data.loader import create_expert_bucket_loaders
            self.expert_loaders = create_expert_bucket_loaders(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Create router data loader
            from data.loader import create_router_loader
            self.router_loader = create_router_loader(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Log initialization statistics
            num_experts = len(self.expert_loaders)
            self.logger.info(f"Created {num_experts} expert data loaders")
            self.logger.info(f"Created router data loader with {len(self.router_loader)} batches")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data loaders: {str(e)}")
            raise
            
    def init_models(self):
        """Initialize router and expert models"""
        self.logger.info("Initializing models")
        
        # Initialize router trainer
        try:
            self.router_trainer = RouterTrainer(
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
            self.logger.info("Router trainer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize router trainer: {str(e)}")
            raise
        
        # Initialize expert trainers - only for experts assigned to this rank
        assigned_experts = []
        for expert_idx in range(self.config.num_experts):
            # Only create experts assigned to this rank
            if expert_idx % self.world_size == self.rank:
                try:
                    expert = ExpertTrainer(
                        expert_idx=expert_idx,
                        config=self.config,
                        device=self.device,
                        rank=self.rank,
                        world_size=self.world_size
                    )
                    self.experts[expert_idx] = expert
                    assigned_experts.append(expert_idx)
                except Exception as e:
                    self.logger.error(f"Failed to initialize expert {expert_idx}: {str(e)}")
                    
        self.logger.info(f"Initialized {len(assigned_experts)} experts on rank {self.rank}: {assigned_experts}")
    
    def get_expert(self, expert_idx):
        """
        Get expert model, loading it from disk if necessary
        
        Args:
            expert_idx: Index of the expert to retrieve
            
        Returns:
            Loaded expert model or None if not owned by this rank
        """
        # Check if expert is owned by this rank
        if not self.is_expert_owned_by_rank(expert_idx):
            return None
            
        # If cache manager is available, use it to efficiently load the expert
        if self.cache_manager is not None:
            # Define expert builder function
            def expert_builder(_):
                logger.info(f"Building expert {expert_idx} for rank {self.rank}")
                return ExpertTrainer(
                    config=self.config,
                    expert_idx=expert_idx,
                    rank=self.rank,
                    world_size=self.world_size
                )
            
            # Use the cache manager to retrieve or load the expert
            return self.cache_manager.get_expert(expert_idx, expert_builder)
        else:
            # No cache manager, create expert directly
            # This is less memory efficient as all experts will stay in memory
            logger.info(f"Building expert {expert_idx} for rank {self.rank} (no cache manager)")
            return ExpertTrainer(
                config=self.config,
                expert_idx=expert_idx,
                rank=self.rank,
                world_size=self.world_size
            )
    
    def perform_reclustering(self):
        """Perform reclustering of the dataset with optimized model migration"""
        self.logger.info("Performing reclustering")
        
        try:
            # Create feature dataset for reclustering
            from data.dataset import FeatureDataset
            feature_dataset = FeatureDataset(
                root_dir=self.config.dataset_path,
                config=self.config
            )
            
            # Create feature loader
            from data.loader import create_loader
            feature_loader = create_loader(
                dataset=feature_dataset,
                config=self.config,
                is_train=False,
                distributed=(self.world_size > 1),
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Perform reclustering
            new_labels, cluster_mapping = self.cluster_manager.perform_reclustering(feature_loader)
            
            # Update dataset clusters
            self.dataset.cluster_assignments = new_labels
            
            # Clear caches to free up memory before migration
            self.cache_manager.clear_cache()
            
            # Recreate per-expert data loaders
            from data.loader import create_expert_bucket_loaders
            self.expert_loaders = create_expert_bucket_loaders(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Create a copy of experts before migration
            old_experts = {idx: expert for idx, expert in self.experts.items()}
            
            # Handle expert parameter migration
            for old_idx, new_idx in cluster_mapping.items():
                if old_idx != new_idx:
                    self.migrate_expert_data(old_idx, new_idx, old_experts)
            
            # Update router data loader to reflect new clustering
            from data.loader import create_router_loader
            self.router_loader = create_router_loader(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            self.logger.info("Reclustering completed successfully")
        except Exception as e:
            self.logger.error(f"Error during reclustering: {str(e)}")
            # Continue with current clustering if reclustering fails
    
    def migrate_expert_data(self, old_idx, new_idx, old_experts):
        """
        Migrate expert data during reclustering
        
        Args:
            old_idx: Old expert index
            new_idx: New expert index
            old_experts: Dictionary of old experts
        """
        self.logger.info(f"Migrating expert data from {old_idx} to {new_idx}")
        
        # Check if we have the source expert
        if old_idx not in old_experts:
            self.logger.warning(f"Source expert {old_idx} not found for migration")
            return
            
        # If we don't own the target expert, no need to do anything
        if new_idx % self.world_size != self.rank:
            self.logger.debug(f"Target expert {new_idx} not assigned to this rank, skipping migration")
            return
            
        try:
            # Get source expert
            source_expert = old_experts[old_idx]
            
            # Create target expert if it doesn't exist
            if new_idx not in self.experts:
                self.experts[new_idx] = ExpertTrainer(
                    expert_idx=new_idx,
                    config=self.config,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size
                )
                
            # Copy parameters from source to target
            target_expert = self.experts[new_idx]
            
            # Copy state dict with proper error handling
            with torch.no_grad():
                source_state = source_expert.expert.state_dict()
                target_expert.expert.load_state_dict(source_state)
                
            self.logger.info(f"Expert data migrated from {old_idx} to {new_idx}")
        except Exception as e:
            self.logger.error(f"Error during expert migration: {str(e)}")
            # Continue with newly initialized expert if migration fails
    
    def train_experts(self, step):
        """
        Train expert models for one step with DFM objective from paper Section 3.2
        
        Args:
            step: Current training step
            
        Returns:
            Average loss across experts
        """
        self.current_step = step
        total_loss = 0.0
        num_experts_trained = 0
        
        # Train each expert assigned to this process
        for expert_idx in sorted(self.expert_loaders.keys()):
            # Skip if not assigned to this rank
            if expert_idx % self.world_size != self.rank:
                continue
                
            # Get expert from cache
            try:
                expert = self.get_expert(expert_idx)
            except Exception as e:
                self.logger.error(f"Failed to get expert {expert_idx}: {str(e)}")
                continue
                
            # Get loader for this expert
            loader = self.expert_loaders.get(expert_idx)
            if not loader:
                continue
                
            # Get batch with error handling
            try:
                # Keep track of iterator
                if not hasattr(self, 'expert_iterators'):
                    self.expert_iterators = {}
                    
                # Get or create iterator
                if expert_idx not in self.expert_iterators:
                    self.expert_iterators[expert_idx] = iter(loader)
                    
                # Get batch, reset iterator if needed
                try:
                    batch = next(self.expert_iterators[expert_idx])
                except StopIteration:
                    self.expert_iterators[expert_idx] = iter(loader)
                    batch = next(self.expert_iterators[expert_idx])
                    
                # Paper Section 3.2: Train expert on its assigned cluster data
                # Each expert specializes on a subset of the data distribution
                loss = expert.train_step(batch)
                total_loss += loss
                num_experts_trained += 1
                
            except Exception as e:
                self.logger.error(f"Error training expert {expert_idx}: {str(e)}")
                continue
        
        # Average loss across experts
        avg_loss = total_loss / max(1, num_experts_trained)
        
        return avg_loss
            
    def train_router(self, step):
        """
        Train router model for one step following Algorithm 1 in the paper
        
        Args:
            step: Current training step
            
        Returns:
            Router loss
        """
        self.current_step = step
        
        # Skip router training if no data loader
        if not hasattr(self, 'router_loader') or not self.router_loader:
            return 0.0
            
        # Get batch with error handling
        try:
            # Keep track of iterator
            if not hasattr(self, 'router_iterator'):
                self.router_iterator = iter(self.router_loader)
                
            # Get batch, reset iterator if needed
            try:
                batch = next(self.router_iterator)
            except StopIteration:
                self.router_iterator = iter(self.router_loader)
                batch = next(self.router_iterator)
                
            # Paper Section 3.3: Train router to predict cluster for each sample
            # Router training follows Algorithm 1 in the paper
            loss = self.router_trainer.train_step(batch)
            
            return loss
        except Exception as e:
            self.logger.error(f"Error training router: {str(e)}")
            return 0.0
    
    def run_ensemble_validation(self, step):
        """
        Run validation using the complete ensemble (paper Section 3.5)
        
        Args:
            step: Current step
        """
        if self.rank != 0:
            return  # Only run on main process
            
        self.logger.info(f"Running ensemble validation at step {step}")
        
        # Setup validation
        num_samples = min(16, self.config.batch_size)  # Small batch for validation
        device = self.device
        
        try:
            # Create validation batch
            val_loader = self.create_validation_loader(batch_size=num_samples)
            batch = next(iter(val_loader))
            
            images = batch["image"].to(device)
            cluster_labels = batch["cluster"].to(device)
            text_embeds = batch.get("text_embedding")
            if text_embeds is not None:
                text_embeds = text_embeds.to(device)
                
            # Create timesteps
            batch_size = images.shape[0]
            t = torch.rand(batch_size, device=device)
            timesteps = (t * 1000).long()
            
            # Forward diffusion
            noise = torch.randn_like(images)
            alpha_t = torch.cos(t.view(-1, 1, 1, 1) * math.pi/2)
            sigma_t = torch.sin(t.view(-1, 1, 1, 1) * math.pi/2)
            noisy_images = alpha_t * images + sigma_t * noise
            
            # Get router predictions
            router_outputs = self.router_trainer.router(noisy_images, timesteps, text_embeds)
            router_probs = torch.softmax(router_outputs, dim=1)
            
            # Load all experts (for validation only)
            all_experts = {}
            for expert_idx in range(self.config.num_experts):
                if expert_idx not in all_experts:
                    # Use cache manager if available
                    if self.cache_manager:
                        all_experts[expert_idx] = self.cache_manager.get_expert(
                            expert_idx,
                            lambda idx: ExpertTrainer(
                                config=self.config,
                                expert_idx=idx,
                                rank=self.rank,
                                world_size=self.world_size
                            )
                        )
                    else:
                        all_experts[expert_idx] = ExpertTrainer(
                            config=self.config,
                            expert_idx=expert_idx,
                            rank=self.rank,
                            world_size=self.world_size
                        )
            
            # Get predictions from all experts
            expert_preds = {}
            for expert_idx, expert in all_experts.items():
                with torch.no_grad():
                    expert_preds[expert_idx] = expert.expert(noisy_images, timesteps, text_embeds)
                    
            # Paper Section 3.4: Combine expert predictions using router weights
            ensemble_pred = torch.zeros_like(noisy_images)
            for expert_idx, pred in expert_preds.items():
                weights = router_probs[:, expert_idx].view(batch_size, 1, 1, 1)
                ensemble_pred += weights * pred
                
            # Compute flow matching target
            target = self.flow_matcher.compute_flow_matching_target(images, noisy_images, t)
            
            # Compute metrics
            ensemble_loss = F.mse_loss(ensemble_pred, target)
            expert_losses = {}
            for expert_idx, pred in expert_preds.items():
                expert_losses[expert_idx] = F.mse_loss(pred, target).item()
                
            # Log metrics
            self.logger.info(f"Ensemble validation loss: {ensemble_loss.item():.6f}")
            for expert_idx, loss in expert_losses.items():
                self.logger.info(f"Expert {expert_idx} validation loss: {loss:.6f}")
                
            # Log router accuracy
            top1_accuracy = (router_probs.argmax(dim=1) == cluster_labels).float().mean()
            self.logger.info(f"Router accuracy: {top1_accuracy.item():.2f}")
            
            # Log metrics to tracking system
            if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
                metrics = {
                    "val/ensemble_loss": ensemble_loss.item(),
                    "val/router_accuracy": top1_accuracy.item(),
                }
                
                for expert_idx, loss in expert_losses.items():
                    metrics[f"val/expert_{expert_idx}_loss"] = loss
                    
                log_metrics(metrics, step=step)
                
        except Exception as e:
            self.logger.error(f"Error in ensemble validation: {str(e)}")
    
    def run_validation(self, step):
        """Run validation for current model state"""
        if not is_main_process():
            return {}
            
        self.logger.info(f"Running validation at step {step}")
        
        # Generate samples
        try:
            samples = self.generate_samples(
                num_samples=self.config.validation_samples,
                guidance_scale=self.config.cfg_scale
            )
            
            # Log sample images
            if samples is not None and len(samples) > 0:
                log_images(
                    images=samples, 
                    step=step,
                    prefix="validation"
                )
                
                # Calculate metrics if configured
                metrics = self.metrics.calculate_metrics(samples, step)
                
                # Log metrics
                log_metrics(metrics, step=step)
                
                return metrics
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            
        return {}
    
    def generate_samples(self, prompts=None, num_samples=4, guidance_scale=7.5):
        """Generate samples from the ensemble model"""
        if not is_main_process():
            return None
            
        # Use default prompts if none provided
        if prompts is None:
            # Use class names for ImageNet or default prompts
            if hasattr(self.config, 'dataset_type') and self.config.dataset_type.lower() == 'imagenet':
                from data.imagenet_classes import IMAGENET_CLASSES
                import random
                prompts = random.sample(IMAGENET_CLASSES, k=min(num_samples, len(IMAGENET_CLASSES)))
            else:
                prompts = [f"sample {i+1}" for i in range(num_samples)]
                
        # Ensure we have enough prompts
        if len(prompts) < num_samples:
            prompts = prompts * (num_samples // len(prompts) + 1)
        prompts = prompts[:num_samples]
        
        # Device to generate on
        device = self.device
        
        # Get router model
        router = self.router_trainer.router
        
        # Get experts
        experts = {}
        for expert_idx in range(self.config.num_experts):
            if expert_idx % self.world_size == self.rank:
                try:
                    experts[expert_idx] = self.get_expert(expert_idx).expert
                except Exception as e:
                    self.logger.error(f"Error loading expert {expert_idx} for sampling: {str(e)}")
                    
        # Create empty list for missing experts
        for expert_idx in range(self.config.num_experts):
            if expert_idx not in experts:
                experts[expert_idx] = None
        
        # Sample shape
        sample_shape = (
            num_samples,
            self.config.latent_channels,
            self.config.image_size // 8,
            self.config.image_size // 8
        )
        
        # Generate samples
        try:
            # Encode prompts
            text_embeddings, uncond_embeddings = self.clip.encode_with_uncond(prompts)
            
            # Sample from diffusion model
            latents = ddm_sample(
                router=router,
                experts=experts,
                shape=sample_shape,
                steps=self.config.inference_steps,
                top_k=self.config.use_top_k,
                device=device,
                cfg_scale=guidance_scale,
                text_embeddings=text_embeddings,
                uncond_embeddings=uncond_embeddings
            )
            
            # Decode latents
            images = self.vae.decode(latents)
            
            # Convert to PIL images
            pil_images = [tensor_to_pil(img) for img in images]
            
            return pil_images
        except Exception as e:
            self.logger.error(f"Error generating samples: {str(e)}")
            return []
    
    def log_sharded_metrics(self, step, expert_loss, router_loss):
        """Log metrics from all shards"""
        if not is_main_process():
            return
            
        metrics = {
            "expert_loss": expert_loss,
            "router_loss": router_loss,
            "step": step
        }
        
        # Log learning rates if available
        if hasattr(self, 'experts') and self.experts:
            expert_idx = next(iter(self.experts.keys()))
            expert = self.experts[expert_idx]
            if hasattr(expert, 'optimizer') and expert.optimizer:
                metrics["expert_lr"] = expert.optimizer.param_groups[0]['lr']
                
        if hasattr(self, 'router_trainer') and hasattr(self.router_trainer, 'optimizer'):
            metrics["router_lr"] = self.router_trainer.optimizer.param_groups[0]['lr']
            
        # Log metrics
        log_metrics(metrics, step=step)
        
    def needs_reclustering(self, step):
        """Check if reclustering should be performed at this step"""
        if not hasattr(self.config, 'recluster_interval'):
            return False
        
        return step > 0 and step % self.config.recluster_interval == 0
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for coordinator"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract step
            step = checkpoint.get('step', 0)
            self.current_step = step
            
            # Extract router state
            router_state = checkpoint.get('router_state', None)
            if router_state and self.router_trainer:
                self.router_trainer.load_state_dict(router_state)
                self.logger.info("Loaded router state from checkpoint")
                
            # Extract expert states
            expert_states = checkpoint.get('expert_states', {})
            for expert_idx, state in expert_states.items():
                # Only load if this expert is assigned to this rank
                if int(expert_idx) % self.world_size == self.rank:
                    if int(expert_idx) not in self.experts:
                        # Create expert if it doesn't exist
                        self.experts[int(expert_idx)] = ExpertTrainer(
                            expert_idx=int(expert_idx),
                            config=self.config,
                            device=self.device,
                            rank=self.rank,
                            world_size=self.world_size
                        )
                    
                    # Load expert state
                    try:
                        self.experts[int(expert_idx)].load_state_dict(state)
                        self.logger.info(f"Loaded state for expert {expert_idx}")
                    except Exception as e:
                        self.logger.error(f"Failed to load state for expert {expert_idx}: {str(e)}")
            
            self.logger.info(f"Checkpoint loaded successfully, resuming from step {step}")
            return step
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return 0
    
    def save_sharded_checkpoints(self, step):
        """Save sharded checkpoints"""
        if not is_main_process():
            return
            
        self.logger.info(f"Saving checkpoint at step {step}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save router
        try:
            router_path = os.path.join(checkpoint_dir, "router.pt")
            torch.save({
                'step': step,
                'state_dict': self.router_trainer.router.state_dict(),
                'optimizer': self.router_trainer.optimizer.state_dict(),
                'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }, router_path)
            self.logger.info(f"Saved router to {router_path}")
        except Exception as e:
            self.logger.error(f"Failed to save router: {str(e)}")
        
        # Save experts
        for expert_idx, expert in self.experts.items():
            try:
                expert_path = os.path.join(checkpoint_dir, f"expert_{expert_idx}.pt")
                torch.save({
                    'step': step,
                    'expert_idx': expert_idx,
                    'state_dict': expert.expert.state_dict(),
                    'optimizer': expert.optimizer.state_dict(),
                    'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
                }, expert_path)
                self.logger.info(f"Saved expert {expert_idx} to {expert_path}")
            except Exception as e:
                self.logger.error(f"Failed to save expert {expert_idx}: {str(e)}")
        
        # Save coordinator checkpoint
        try:
            coordinator_path = os.path.join(checkpoint_dir, "coordinator.pt")
            torch.save({
                'step': step,
                'router_state': self.router_trainer.state_dict() if self.router_trainer else None,
                'expert_states': {idx: expert.state_dict() for idx, expert in self.experts.items()},
                'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }, coordinator_path)
            self.logger.info(f"Saved coordinator checkpoint to {coordinator_path}")
        except Exception as e:
            self.logger.error(f"Failed to save coordinator checkpoint: {str(e)}")
    
    def train_distilled_model(self):
        """Train distilled model from experts"""
        self.logger.info("Starting model distillation")
        
        try:
            # Initialize distiller
            distiller = DiffusionDistiller(
                config=self.config,
                experts={idx: expert.expert for idx, expert in self.experts.items()},
                router=self.router_trainer.router,
                dataset=self.dataset,
                device=self.device,
                rank=self.rank
            )
            
            # Train distilled model
            distiller.train()
            
            # Save distilled model
            distill_path = os.path.join(self.config.checkpoint_dir, "distilled_model.pt")
            distiller.save(distill_path)
            
            self.logger.info(f"Distilled model saved to {distill_path}")
        except Exception as e:
            self.logger.error(f"Error during distillation: {str(e)}")
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cache_manager'):
            try:
                self.cache_manager.shutdown()
            except:
                pass

    def is_expert_owned_by_rank(self, expert_idx):
        """
        Check if an expert is owned by this rank
        
        Args:
            expert_idx: Index of the expert to check
            
        Returns:
            True if expert is owned by this rank, False otherwise
        """
        # Simple sharding: expert_idx % world_size == rank
        return expert_idx % self.world_size == self.rank
