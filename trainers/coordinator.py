"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""

import math
import torch
import os
import datetime
import torch.nn.functional as F
import time
import sys

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

# Direct console print function for immediate feedback
def debug_print(message, rank=None, force=False):
    """Print directly to console regardless of logger configuration"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if rank is not None:
        prefix = f"[COORD-{rank}]"
    else:
        prefix = "[COORD]"
        
    if force or (rank is not None and rank == 0) or rank is None:
        print(f"{prefix} [{timestamp}] {message}", flush=True)

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
        init_start_time = time.time()
        debug_print(f"Starting coordinator initialization on rank {rank}/{world_size}", rank)
        
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.cache_manager = cache_manager
        
        debug_print(f"Initializing with device: {self.device}", rank)
        
        # Initialize clustering
        debug_print(f"Starting cluster manager initialization", rank)
        cluster_start = time.time()
        self.init_cluster_manager()
        debug_print(f"Cluster manager initialized in {time.time() - cluster_start:.2f}s", rank)
        
        # Initialize data loaders
        debug_print(f"Starting data loader initialization", rank)
        data_start = time.time()
        self.init_data_loaders()
        debug_print(f"Data loaders initialized in {time.time() - data_start:.2f}s", rank)
        
        # Initialize models
        debug_print(f"Starting model initialization", rank)
        models_start = time.time()
        self.init_models()
        debug_print(f"Models initialized in {time.time() - models_start:.2f}s", rank)
        
        debug_print(f"Setting up optimizers and schedulers", rank)
        
        # Set up optimizers and schedulers
        self.optimizers = {}
        self.schedulers = {}
        
        # Initialize router optimizer
        debug_print(f"Initializing router optimizer", rank)
        self.optimizers['router'] = torch.optim.AdamW(
            self.router.parameters(),
            lr=config.router_learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # Initialize expert optimizers (one per expert)
        debug_print(f"Initializing expert optimizers", rank)
        for expert_idx in range(config.num_experts):
            if self.is_expert_owned_by_rank(expert_idx):
                # Only create optimizer for experts owned by this rank
                debug_print(f"Creating optimizer for expert {expert_idx} on rank {rank}", rank)
                expert = self.get_expert(expert_idx)
                self.optimizers[f'expert_{expert_idx}'] = torch.optim.AdamW(
                    expert.parameters(),
                    lr=config.learning_rate,
                    betas=config.adam_betas,
                    weight_decay=config.weight_decay
                )
        
        # Create learning rate schedulers (cosine decay)
        debug_print(f"Setting up learning rate schedulers", rank)
        warmup_steps = getattr(config, 'warmup_steps', 1000)
        total_steps = config.num_steps
        
        for key, optimizer in self.optimizers.items():
            self.schedulers[key] = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(step / warmup_steps, 1.0) if step < warmup_steps else 
                0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            )
        
        # Initialize flow matcher for loss computation
        debug_print(f"Initializing flow matcher", rank)
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma,
            loss_type=config.loss_type
        )
        
        # Get diffusion parameters
        debug_print(f"Computing diffusion parameters", rank)
        self.alphas, self.alpha_bar, _ = get_alphas_and_betas()
        
        # Track metrics
        self.metrics = {
            'step': 0,
            'expert_loss': {},
            'router_loss': 0.0,
            'learning_rates': {}
        }
        
        # Create AMP grad scaler for mixed precision training
        debug_print(f"Setting up mixed precision training", rank)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
        
        # Expert data loaders and iterators
        self.expert_iterators = {}
        
        total_init_time = time.time() - init_start_time
        debug_print(f"DDMTrainingCoordinator initialization completed in {total_init_time:.2f}s", rank)
        logger.info(f"DDMTrainingCoordinator initialized on rank {rank}/{world_size}")
        
        # If we're the main process, log the configuration
        if self.rank == 0:
            logger.info(f"Training configuration:")
            for key, value in vars(config).items():
                if not key.startswith('_'):
                    logger.info(f"  {key}: {value}")
                    
        # Synchronize to ensure all processes have completed initialization
        debug_print(f"Waiting for synchronization across all processes", rank)
        sync_start = time.time()
        synchronize()
        debug_print(f"All processes synchronized in {time.time() - sync_start:.2f}s", rank)
        
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
        debug_print(f"Creating ClusterManager for rank {self.rank}", self.rank)
        
        try:
            # Create cluster manager
            self.cluster_manager = ClusterManager(
                local_rank=self.rank,
                dataset_path=self.config.dataset_path,
                config=self.config
            )
            
            # Use synchronization to ensure all processes have initialized
            debug_print(f"Waiting for cluster manager sync on rank {self.rank}", self.rank)
            sync_start = time.time()
            self.safe_synchronize(timeout_seconds=300, name="cluster_manager_init")
            debug_print(f"Cluster manager sync completed in {time.time() - sync_start:.2f}s", self.rank)
            
            debug_print(f"Cluster manager initialized successfully on rank {self.rank}", self.rank)
        except Exception as e:
            debug_print(f"CRITICAL ERROR: Failed to initialize cluster manager on rank {self.rank}: {str(e)}", self.rank, True)
            logger.error(f"Failed to initialize cluster manager: {str(e)}")
            raise
            
    def init_data_loaders(self):
        """Initialize data loaders for training"""
        debug_print(f"Initializing data loaders on rank {self.rank}", self.rank)
        
        try:
            # Create dataset
            debug_print(f"Creating dataset on rank {self.rank}", self.rank)
            self.dataset = DDMDataset(
                root_dir=self.config.dataset_path,
                cluster_assignments=self.cluster_manager.get_cluster_assignments(),
                config=self.config,
                vae=self.vae,
                clip=self.clip
            )
            
            # Create expert data loaders
            debug_print(f"Creating expert bucket loaders on rank {self.rank}", self.rank)
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
        """Initialize models for training"""
        debug_print(f"Initializing models on rank {self.rank}", self.rank)
        
        try:
            # Create VAE for latent encoding
            debug_print(f"Creating VAE encoder/decoder on rank {self.rank}", self.rank)
            start_time = time.time()
            from data.vae import VAEWrapper
            self.vae = VAEWrapper(self.device, self.config)
            debug_print(f"VAE created in {time.time() - start_time:.2f}s on rank {self.rank}", self.rank)
            
            # Create CLIP for text conditioning
            debug_print(f"Creating CLIP encoder on rank {self.rank}", self.rank)
            start_time = time.time()
            from data.clip import CLIPTextEncoder
            self.clip = CLIPTextEncoder(self.device, self.config)
            debug_print(f"CLIP created in {time.time() - start_time:.2f}s on rank {self.rank}", self.rank)
            
            # Create router trainer
            debug_print(f"Creating router model on rank {self.rank}", self.rank)
            start_time = time.time()
            self.router = RouterTrainer(
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
            debug_print(f"Router created in {time.time() - start_time:.2f}s on rank {self.rank}", self.rank)
            
            # Create expert trainers (one per expert)
            debug_print(f"Creating expert models for rank {self.rank}", self.rank)
            self.experts = {}
            
            # Only create expert models that are assigned to this rank
            for expert_idx in range(self.config.num_experts):
                if self.is_expert_owned_by_rank(expert_idx):
                    debug_print(f"Creating expert {expert_idx} on rank {self.rank}", self.rank)
                    start_time = time.time()
                    if self.cache_manager:
                        # If using cache manager, get expert from cache or create new
                        self.experts[expert_idx] = self.cache_manager.get_expert(expert_idx)
                    else:
                        # Otherwise create expert directly
                        self.experts[expert_idx] = ExpertTrainer(
                            expert_idx=expert_idx,
                            config=self.config,
                            device=self.device,
                            rank=self.rank,
                            world_size=self.world_size
                        )
                    debug_print(f"Expert {expert_idx} created in {time.time() - start_time:.2f}s on rank {self.rank}", self.rank)
            
            debug_print(f"Model initialization complete on rank {self.rank}", self.rank)
            
        except Exception as e:
            debug_print(f"ERROR: Failed to initialize models on rank {self.rank}: {str(e)}", self.rank, True)
            logger.error(f"Failed to initialize models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        # Use synchronization
        debug_print(f"Synchronizing after model initialization on rank {self.rank}", self.rank)
        sync_start = time.time()
        self.safe_synchronize(timeout_seconds=300, name="model_init")
        debug_print(f"Model initialization synchronization completed in {time.time() - sync_start:.2f}s on rank {self.rank}", self.rank)
    
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
        """
        Perform reclustering as described in paper Section 3.6
        
        This updates data partitions based on current model performance
        and reassigns experts to clusters.
        """
        self.logger.info("Starting reclustering procedure")
        
        # Step 1: Extract features from validation set
        features = self.cluster_manager.extract_validation_features()
        
        # Step 2: Perform clustering on extracted features
        old_clusters = self.cluster_manager.get_cluster_assignments()
        new_clusters = self.cluster_manager.update_clusters(features)
        
        # Step 3: Compute cluster changes
        changed_ratio = self.cluster_manager.compute_cluster_changes(old_clusters, new_clusters)
        self.logger.info(f"Reclustering: {changed_ratio:.2%} of data points changed clusters")
        
        # Step 4: Update data loaders with new clusters
        self.dataset.update_cluster_assignments(new_clusters)
        
        # Step 5: Recreate expert data loaders with updated clusters
        old_expert_loaders = self.expert_loaders
        from data.loader import create_expert_bucket_loaders
        self.expert_loaders = create_expert_bucket_loaders(
            dataset=self.dataset,
            config=self.config,
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Step 6: Reset expert iterators
        self.expert_iterators = {}
        
        # Step 7: Migrate expert weights based on cluster similarity if enabled
        if getattr(self.config, 'expert_migration_strategy', 'reset') == 'migrate':
            # Get old experts
            old_experts = {}
            for expert_idx in range(self.config.num_experts):
                if self.is_expert_owned_by_rank(expert_idx):
                    old_experts[expert_idx] = self.get_expert(expert_idx)
            
            # Compute similarity matrix between old and new clusters
            similarity_matrix = self.cluster_manager.compute_cluster_similarity(old_clusters, new_clusters)
            
            # For each new expert, find the most similar old expert
            for new_idx in range(self.config.num_experts):
                if self.is_expert_owned_by_rank(new_idx):
                    # Find the most similar old expert
                    similarities = similarity_matrix[:, new_idx]
                    old_idx = similarities.argmax().item()
                    
                    # Migrate expert weights
                    self.migrate_expert_data(old_idx, new_idx, old_experts)
        else:
            # Reset expert weights
            for expert_idx in range(self.config.num_experts):
                if self.is_expert_owned_by_rank(expert_idx):
                    expert = self.get_expert(expert_idx)
                    expert.reset_parameters()
        
        # Step 8: Synchronize after reclustering is complete
        self.safe_synchronize(timeout_seconds=300, name="reclustering")
        
        self.logger.info("Reclustering completed successfully")
        return True
    
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
        Train expert models according to paper Section 3.2 and 3.4
        
        Args:
            step: Current training step
            
        Returns:
            Mean expert loss
        """
        expert_losses = {}
        
        # Each process trains its assigned experts
        for expert_idx in range(self.config.num_experts):
            if self.is_expert_owned_by_rank(expert_idx):
                # Get expert data loader iterator
                if expert_idx not in self.expert_iterators:
                    # Initialize iterator if not exists
                    self.expert_iterators[expert_idx] = iter(self.expert_loaders[expert_idx])
                
                try:
                    # Get next batch
                    batch = next(self.expert_iterators[expert_idx])
                except StopIteration:
                    # Reset iterator and get new batch
                    self.expert_iterators[expert_idx] = iter(self.expert_loaders[expert_idx])
                    batch = next(self.expert_iterators[expert_idx])
                
                # Get expert model
                expert = self.get_expert(expert_idx)
                # Set to training mode
                expert.train()
                
                # Train step (Section 3.2 of the paper)
                loss = expert.train_step(batch)
                expert_losses[expert_idx] = loss
                
                # Log per-expert metrics
                if step % self.config.log_every_n_steps == 0 and self.rank == 0:
                    self.logger.info(f"Step {step}: Expert {expert_idx} loss: {loss:.4f}")
        
        # Synchronize expert losses across processes
        all_expert_losses = {}
        for i in range(self.config.num_experts):
            if i in expert_losses:
                all_expert_losses[i] = expert_losses[i]
            else:
                all_expert_losses[i] = 0.0
        
        # Calculate mean loss
        mean_loss = sum(all_expert_losses.values()) / max(len(all_expert_losses), 1)
        
        return mean_loss
            
    def train_router(self, step):
        """
        Train router model according to paper Section 3.3 (Algorithm 1)
        
        Args:
            step: Current training step
            
        Returns:
            Router loss
        """
        # Only train router on rank 0
        if self.rank != 0:
            return 0.0
        
        # Get router data loader iterator
        if not hasattr(self, 'router_iterator'):
            self.router_iterator = iter(self.router_loader)
        
        try:
            # Get next batch
            batch = next(self.router_iterator)
        except StopIteration:
            # Reset iterator and get new batch
            self.router_iterator = iter(self.router_loader)
            batch = next(self.router_iterator)
        
        # Train step (Section 3.3 of the paper)
        loss = self.router.train_step(batch)
        
        # Log router metrics
        if step % self.config.log_every_n_steps == 0:
            self.logger.info(f"Step {step}: Router loss: {loss:.4f}")
        
        return loss
    
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
            router_outputs = self.router.router(noisy_images, timesteps, text_embeds)
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
        router = self.router.router
        
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
                
        if hasattr(self, 'router') and hasattr(self.router, 'optimizer'):
            metrics["router_lr"] = self.router.optimizer.param_groups[0]['lr']
            
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
            if router_state and self.router:
                self.router.load_state_dict(router_state)
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
                'state_dict': self.router.state_dict(),
                'optimizer': self.router.optimizer.state_dict(),
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
                    'state_dict': expert.state_dict(),
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
                'router_state': self.router.state_dict() if self.router else None,
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
                router=self.router,
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
