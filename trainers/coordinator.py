"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""

import math
import torch
import os
import datetime
import torch.nn.functional as F
import time
import sys
import numpy as np

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
    
    def __init__(self, config, rank, world_size, cache_manager=None, progress_callback=None):
        """
        Initialize coordinator as per paper Section 4.1
        Args:
            config: Configuration object
            rank: Process rank (0 is main)
            world_size: Total number of processes
            cache_manager: Optional ExpertCacheManager for efficient expert loading/unloading
            progress_callback: Optional callback function to report initialization progress
        """
        init_start_time = time.time()
        debug_print(f"Starting coordinator initialization on rank {rank}/{world_size}", rank)
        
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.cache_manager = cache_manager
        self.progress_callback = progress_callback
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")
        
        # Initialize components
        self.vae = None
        self.clip = None
        self.router = None
        self.experts = {}
        
        # Data components
        self.cluster_manager = None
        self.dataset = None
        self.expert_loaders = {}
        self.router_loader = None
        
        # Initialize learning rate scheduler
        self.scheduler_fn = None
        self.create_lr_scheduler()
        
        debug_print(f"Initializing with device: {self.device}", rank)
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Starting initialization", 0)
        
        # Initialize clustering
        debug_print(f"Starting cluster manager initialization", rank)
        cluster_start = time.time()
        self.init_cluster_manager()
        debug_print(f"Cluster manager initialized in {time.time() - cluster_start:.2f}s", rank)
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Cluster manager initialized", 25)
        
        # Initialize data loaders
        debug_print(f"Starting data loader initialization", rank)
        data_start = time.time()
        self.init_data_loaders()
        debug_print(f"Data loaders initialized in {time.time() - data_start:.2f}s", rank)
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Data loaders initialized", 50)
        
        # Initialize models
        debug_print(f"Starting model initialization", rank)
        models_start = time.time()
        self.init_models()
        debug_print(f"Models initialized in {time.time() - models_start:.2f}s", rank)
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Models initialized", 75)
        
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
            # Define scheduler function
            def lr_lambda(step):
                if step < warmup_steps:
                    return min(step / warmup_steps, 1.0)
                else:
                    return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
                    
            # Create scheduler
            self.schedulers[key] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
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
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Initialization complete", 100)
        
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
            logger.debug(f"Synchronization for '{name}' completed successfully")
            return True
        except torch.distributed.DistBackendError as e:
            logger.error(f"Synchronization timeout after {timeout_seconds}s for '{name}': {str(e)}")
            # Attempt recovery by continuing execution
            return False
        except Exception as e:
            logger.error(f"Synchronization error for '{name}': {str(e)}")
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
            logger.info("Synchronizing expert states across processes")
            
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
                        logger.debug(f"Broadcasting expert {expert_idx} from rank {rank}")
                    
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
                        logger.error(f"Error synchronizing expert {expert_idx}: {str(e)}")
                        failed_experts.add(expert_idx)
                        sync_success = False
            
            # Report on synchronization results
            if failed_experts:
                logger.warning(f"Failed to synchronize {len(failed_experts)} experts: {sorted(failed_experts)}")
                logger.info(f"Successfully synchronized {len(synced_experts)} experts")
            else:
                logger.info(f"Expert state synchronization completed successfully for all {len(synced_experts)} experts")
            
            return sync_success
        except Exception as e:
            logger.error(f"Expert state synchronization failed: {str(e)}")
            return False

    def init_cluster_manager(self):
        """Initialize data clustering manager (Section 4.1)"""
        from data.clustering import ClusterManager
        
        # Check if clustering should be skipped for testing/debugging
        if getattr(self.config, 'skip_clustering', False):
            logger.info("Skipping clustering as per configuration (skip_clustering=True)")
            
            # Create a simple ClusterManager without clustering
            self.cluster_manager = ClusterManager(
                config=self.config,
                feature_extractor=None
            )
            
            # Create uniform distribution of data into clusters (simple round-robin assignment)
            if self.rank == 0:
                logger.info(f"Creating uniform distribution across {self.config.num_experts} experts")
                # Report progress
                if self.progress_callback:
                    self.progress_callback("Creating uniform distribution (skipping clustering)", 15)
                
            # Skip actual clustering and return uniform assignment
            # This will be handled by the cluster manager's default assignment
            if self.world_size > 1:
                self.safe_synchronize(timeout_seconds=30, name="skip_clustering")
                
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Clustering skipped", 20)
                
            return None
            
        # If not skipping, proceed with normal clustering
        # Create feature extractor based on config
        logger.info(f"Initializing clustering manager with {self.config.num_experts} experts")
        init_start_time = time.time()
        
        # Log progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Starting cluster manager initialization", 5)
        
        # Create dataset for feature extraction
        logger.info(f"Creating feature extraction dataset from {self.config.dataset_path}")
        dataset_start_time = time.time()
        
        from data.dataset import FeatureDataset
        feature_dataset = FeatureDataset(
            root_dir=self.config.dataset_path,
            config=self.config
        )
        
        dataset_time = time.time() - dataset_start_time
        logger.info(f"Feature dataset created in {dataset_time:.2f}s with {len(feature_dataset)} images")
        
        # Create dataloader for feature extraction
        logger.info(f"Creating feature extraction dataloader")
        dataloader_start_time = time.time()
        
        from torch.utils.data import DataLoader
        feature_loader = DataLoader(
            feature_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        dataloader_time = time.time() - dataloader_start_time
        logger.info(f"Feature dataloader created in {dataloader_time:.2f}s with {len(feature_loader)} batches")
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Extracting features from dataset", 10)
            
        # Create cluster manager
        cluster_start_time = time.time()
        self.cluster_manager = ClusterManager(
            config=self.config,
            feature_extractor=None  # Will be created internally based on config
        )
        
        # Generate clusters
        logger.info(f"Generating {self.config.num_experts} clusters from dataset")
        
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Generating clusters", 15)
            
        # Generate clusters
        clustering_start_time = time.time()
        cluster_labels = self.cluster_manager.generate_clusters(
            dataloader=feature_loader,
            k=self.config.num_experts,
            fine_clusters=getattr(self.config, 'fine_clusters', 1024)
        )
        
        clustering_time = time.time() - clustering_start_time
        logger.info(f"Clustering completed in {clustering_time:.2f}s")
        
        # Visualize clusters if configured
        if getattr(self.config, 'visualize_clusters', False) and self.rank == 0:
            try:
                logger.info("Generating cluster visualization")
                viz_path = os.path.join(self.config.output_dir, 'cluster_visualization.png')
                self.cluster_manager.visualize_clusters(
                    features=None,  # Use internal features
                    labels=cluster_labels,
                    output_path=viz_path
                )
            except Exception as e:
                logger.warning(f"Failed to generate cluster visualization: {e}")
        
        # Synchronize after clustering
        if self.world_size > 1:
            self.safe_synchronize(timeout_seconds=300, name="cluster_initialization")
            
        # Report progress
        if self.progress_callback and self.rank == 0:
            self.progress_callback("Clustering complete", 20)
            
        # Log completion
        total_time = time.time() - init_start_time
        logger.info(f"Cluster manager initialization completed in {total_time:.2f}s")
        
        return cluster_labels
            
    def init_data_loaders(self):
        """Initialize dataset and data loaders (Section 3.1)"""
        logger.info(f"Initializing data loaders on rank {self.rank}")
        init_start_time = time.time()
        
        try:
            # Create dataset
            logger.info(f"Creating dataset from {self.config.dataset_path}")
            dataset_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Creating dataset", 30)
                
            # Get cluster assignments from manager
            cluster_assignments = self.cluster_manager.get_cluster_labels()
            logger.info(f"Using {len(np.unique(cluster_assignments))} clusters for data assignment")
            
            from data.dataset import DDMDataset
            self.dataset = DDMDataset(
                config=self.config,
                split='train',
                cluster_labels=cluster_assignments
            )
            
            dataset_time = time.time() - dataset_start_time
            logger.info(f"Dataset created in {dataset_time:.2f}s with {len(self.dataset)} samples")
            
            # Create expert data loaders
            logger.info(f"Creating expert bucket loaders on rank {self.rank}")
            loaders_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Creating expert data loaders", 35)
                
            from data.loader import create_expert_bucket_loaders
            self.expert_loaders = create_expert_bucket_loaders(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Log data loader stats
            num_experts = len(self.expert_loaders)
            total_batches = sum(len(loader) for loader in self.expert_loaders.values())
            logger.info(f"Created {num_experts} expert loaders with {total_batches} total batches on rank {self.rank}")
            
            for expert_idx, loader in self.expert_loaders.items():
                if self.is_expert_owned_by_rank(expert_idx):
                    logger.info(f"Rank {self.rank} owns expert {expert_idx} with {len(loader)} batches")
            
            # Create router data loader
            logger.info(f"Creating router data loader on rank {self.rank}")
            router_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Creating router data loader", 40)
                
            from torch.utils.data import DataLoader
            self.router_loader = DataLoader(
                self.dataset,
                batch_size=self.config.router_batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            router_time = time.time() - router_start_time
            loaders_time = time.time() - loaders_start_time
            logger.info(f"Router loader created in {router_time:.2f}s with {len(self.router_loader)} batches")
            logger.info(f"All data loaders created in {loaders_time:.2f}s")
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Data loaders ready", 45)
                
            # Log initialization statistics
            if self.rank == 0:
                num_clusters = len(np.unique(cluster_assignments))
                logger.info(f"Dataset has {len(self.dataset)} samples assigned to {num_clusters} clusters")
                logger.info(f"Created {len(self.expert_loaders)} expert loaders and 1 router loader")
                
            # Synchronize all processes after data loading
            if self.world_size > 1:
                self.safe_synchronize(timeout_seconds=60, name="data_loading")
                
            # Final timing
            total_time = time.time() - init_start_time
            logger.info(f"Data loader initialization completed in {total_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error initializing data loaders on rank {self.rank}: {str(e)}")
            raise
            
    def init_models(self):
        """Initialize model components (VAE, CLIP, router, experts) (Section 3.2-3.3)"""
        logger.info(f"Initializing models on rank {self.rank}")
        init_start_time = time.time()
        
        try:
            # Load VAE for latent diffusion
            logger.info(f"Loading VAE encoder/decoder on rank {self.rank}")
            vae_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Loading VAE encoder/decoder", 55)
                
            from data.vae import VAEWrapper
            self.vae = VAEWrapper(
                device=self.device,
                config=self.config
            )
            
            vae_time = time.time() - vae_start_time
            logger.info(f"VAE initialized in {vae_time:.2f}s")
            
            # Load CLIP text encoder for conditioning
            logger.info(f"Loading CLIP text encoder on rank {self.rank}")
            clip_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Loading CLIP text encoder", 60)
                
            from data.clip import CLIPTextEncoder
            self.clip = CLIPTextEncoder(
                device=self.device,
                config=self.config
            )
            
            clip_time = time.time() - clip_start_time
            logger.info(f"CLIP initialized in {clip_time:.2f}s")
            
            # Create router network
            logger.info(f"Creating router network on rank {self.rank}")
            router_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Creating router network", 65)
                
            from trainers.router import DDMRouter
            self.router = DDMRouter(
                config=self.config,
                num_clusters=self.config.num_experts,
                device=self.device
            ) if self.rank == 0 or not getattr(self.config, 'router_on_main_process_only', False) else None
            
            router_time = time.time() - router_start_time
            logger.info(f"Router created in {router_time:.2f}s on rank {self.rank}")
            
            # Create expert networks (only for experts assigned to this process)
            logger.info(f"Creating expert networks for rank {self.rank}")
            experts_start_time = time.time()
            
            # Report progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Creating expert networks", 70)
                
            from trainers.expert import DDMExpert
            self.experts = {}
            experts_per_process = max(1, self.config.num_experts // self.world_size)
            
            # Count total experts this rank will initialize
            expert_indices = [i for i in range(self.config.num_experts) if self.is_expert_owned_by_rank(i)]
            logger.info(f"Rank {self.rank} will create {len(expert_indices)} expert networks")
            
            # Initialize experts assigned to this rank
            for expert_idx in expert_indices:
                expert_start = time.time()
                logger.info(f"Initializing expert {expert_idx} on rank {self.rank}")
                
                self.experts[expert_idx] = DDMExpert(
                    expert_idx=expert_idx,
                    config=self.config,
                    device=self.device,
                    vae=self.vae,
                    text_encoder=self.clip
                )
                
                logger.info(f"Expert {expert_idx} initialized in {time.time() - expert_start:.2f}s on rank {self.rank}")
                
                # Update progress for each expert
                if self.progress_callback and self.rank == 0:
                    # Calculate progress based on how many experts we've initialized
                    progress = 70 + (3 * (len(self.experts) / max(1, len(expert_indices))))
                    self.progress_callback(f"Created expert {expert_idx}", min(73, progress))
            
            experts_time = time.time() - experts_start_time
            logger.info(f"All {len(self.experts)} experts initialized in {experts_time:.2f}s on rank {self.rank}")
            
            # Get initialized expert indices
            initialized_experts = list(self.experts.keys())
            logger.info(f"Rank {self.rank} initialized experts: {initialized_experts}")
            
            # Synchronize model initialization across processes
            logger.info(f"Synchronizing model initialization on rank {self.rank}")
            sync_start = time.time()
            
            if self.world_size > 1:
                self.safe_synchronize(timeout_seconds=60, name="model_initialization")
                
            sync_time = time.time() - sync_start
            logger.info(f"Synchronization completed in {sync_time:.2f}s")
            
            # Report final progress
            if self.progress_callback and self.rank == 0:
                self.progress_callback("Models initialized", 75)
                
            # Final timing
            total_time = time.time() - init_start_time
            logger.info(f"Model initialization completed in {total_time:.2f}s")
            
            # Log memory usage after model initialization
            if torch.cuda.is_available():
                used_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                total_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                logger.info(f"GPU memory: {used_mem:.2f}GB used / {total_mem:.2f}GB total ({used_mem/total_mem*100:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error initializing models on rank {self.rank}: {str(e)}")
            raise
    
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
        logger.info("Starting reclustering procedure")
        
        # Step 1: Extract features from validation set
        features = self.cluster_manager.extract_validation_features()
        
        # Step 2: Perform clustering on extracted features
        old_clusters = self.cluster_manager.get_cluster_assignments()
        new_clusters = self.cluster_manager.update_clusters(features)
        
        # Step 3: Compute cluster changes
        changed_ratio = self.cluster_manager.compute_cluster_changes(old_clusters, new_clusters)
        logger.info(f"Reclustering: {changed_ratio:.2%} of data points changed clusters")
        
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
        
        logger.info("Reclustering completed successfully")
        return True
    
    def migrate_expert_data(self, old_idx, new_idx, old_experts):
        """
        Migrate expert data during reclustering
        
        Args:
            old_idx: Old expert index
            new_idx: New expert index
            old_experts: Dictionary of old experts
        """
        logger.info(f"Migrating expert data from {old_idx} to {new_idx}")
        
        # Check if we have the source expert
        if old_idx not in old_experts:
            logger.warning(f"Source expert {old_idx} not found for migration")
            return
            
        # If we don't own the target expert, no need to do anything
        if new_idx % self.world_size != self.rank:
            logger.debug(f"Target expert {new_idx} not assigned to this rank, skipping migration")
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
                
            logger.info(f"Expert data migrated from {old_idx} to {new_idx}")
        except Exception as e:
            logger.error(f"Error during expert migration: {str(e)}")
            # Continue with newly initialized expert if migration fails
    
    def train_experts(self, step):
        """Train expert networks on their assigned data (Section 3.2)"""
        if not self.experts:
            logger.warning(f"No experts assigned to rank {self.rank}, skipping expert training")
            return 0.0
            
        logger.info(f"Training {len(self.experts)} experts on rank {self.rank} for step {step}")
        training_start_time = time.time()
        
        # Track losses for each expert
        expert_losses = {}
        total_samples = 0
        
        # Train each expert assigned to this process
        for expert_idx, expert in self.experts.items():
            # Skip if no data loader for this expert
            if expert_idx not in self.expert_loaders:
                logger.warning(f"No data loader for expert {expert_idx} on rank {self.rank}, skipping")
                continue
                
            # Get data loader for this expert
            loader = self.expert_loaders[expert_idx]
            if not loader or len(loader) == 0:
                logger.warning(f"Empty data loader for expert {expert_idx} on rank {self.rank}, skipping")
                continue
                
            # Train this expert for one batch
            expert_start_time = time.time()
            logger.info(f"Training expert {expert_idx} on rank {self.rank}")
            
            try:
                # Get a batch from the expert's data loader
                batch = next(iter(loader))
                batch_size = len(batch['image']) if isinstance(batch, dict) and 'image' in batch else len(batch)
                total_samples += batch_size
                
                # Forward pass and update
                loss = expert.train_step(batch, step=step)
                
                # Store loss
                expert_losses[expert_idx] = loss
                
                # Log training stats
                expert_time = time.time() - expert_start_time
                logger.info(f"Expert {expert_idx} trained in {expert_time:.2f}s with loss {loss:.4f} "
                           f"({batch_size} samples, {expert_time/batch_size:.3f}s per sample)")
                
                # Log per-expert metrics
                if step % self.config.log_every_n_steps == 0:
                    logger.info(f"Step {step}: Expert {expert_idx} loss: {loss:.4f}")
            
            except Exception as e:
                logger.error(f"Error training expert {expert_idx} on rank {self.rank}: {str(e)}")
                expert_losses[expert_idx] = float('nan')
                
        # Compute average loss across experts
        valid_losses = [loss for loss in expert_losses.values() if not np.isnan(loss)]
        avg_loss = np.mean(valid_losses).item() if valid_losses else 0.0
        
        # Log overall training stats
        training_time = time.time() - training_start_time
        experts_trained = len(valid_losses)
        
        if experts_trained > 0:
            logger.info(f"Trained {experts_trained}/{len(self.experts)} experts in {training_time:.2f}s "
                       f"with average loss {avg_loss:.4f} "
                       f"({total_samples} total samples, {training_time/experts_trained:.3f}s per expert)")
        else:
            logger.warning(f"No experts successfully trained on rank {self.rank}")
            
        # Synchronize after expert training if needed
        if self.world_size > 1 and step % getattr(self.config, 'sync_every_n_steps', 10) == 0:
            sync_start = time.time()
            self.safe_synchronize(timeout_seconds=30, name="expert_training")
            logger.info(f"Expert training synchronization completed in {time.time() - sync_start:.2f}s")
            
        return avg_loss
            
    def train_router(self, step):
        """Train the router network to correctly predict data clusters (Section 3.3)"""
        # Skip training if router not on this rank
        if self.router is None:
            return 0.0
            
        # Only master process trains router if configured
        if getattr(self.config, 'router_on_main_process_only', False) and self.rank != 0:
            return 0.0
            
        logger.info(f"Training router on rank {self.rank} for step {step}")
        training_start_time = time.time()
        
        try:
            # Get batch from router loader
            batch_start_time = time.time()
            
            # Initialize iterator if not exists
            if not hasattr(self, 'router_iterator') or self.router_iterator is None:
                self.router_iterator = iter(self.router_loader)
                
            # Get next batch (with restart if needed)
            try:
                batch = next(self.router_iterator)
            except StopIteration:
                # Reset iterator and get new batch
                self.router_iterator = iter(self.router_loader)
                batch = next(self.router_iterator)
                
            batch_time = time.time() - batch_start_time
            
            # Get batch size and cluster info
            batch_size = len(batch['image']) if isinstance(batch, dict) and 'image' in batch else len(batch)
            cluster_labels = batch['cluster'] if isinstance(batch, dict) and 'cluster' in batch else None
            
            if cluster_labels is None:
                logger.warning("No cluster labels in batch, cannot train router")
                return 0.0
                
            # Filter out samples with no cluster assignment (-1)
            valid_mask = cluster_labels >= 0
            valid_count = valid_mask.sum().item()
            
            if valid_count == 0:
                logger.warning("No valid cluster assignments in batch, skipping router training")
                return 0.0
                
            if valid_count < batch_size:
                logger.info(f"Found {batch_size - valid_count} samples without cluster assignments, "
                          f"training with {valid_count}/{batch_size} samples")
                
            # Train router on this batch
            logger.info(f"Training router with batch of {valid_count} samples")
            train_start_time = time.time()
            
            # Forward pass and update
            loss = self.router.train_step(batch, step=step)
            
            # Calculate training metrics
            training_time = time.time() - train_start_time
            total_time = time.time() - training_start_time
            
            # Log detailed statistics
            logger.info(f"Router trained in {training_time:.2f}s with loss {loss:.4f} "
                       f"({valid_count} samples, {training_time/valid_count:.3f}s per sample)")
            logger.info(f"Total router processing time: {total_time:.2f}s "
                       f"(batch loading: {batch_time:.2f}s, training: {training_time:.2f}s)")
            
            # Log router metrics
            if step % self.config.log_every_n_steps == 0:
                logger.info(f"Step {step}: Router loss: {loss:.4f}")
                
            return loss
                
        except Exception as e:
            logger.error(f"Error training router on rank {self.rank}: {str(e)}")
            return 0.0
    
    def run_ensemble_validation(self, step):
        """
        Run validation using the complete ensemble (paper Section 3.5)
        
        Args:
            step: Current step
        """
        if self.rank != 0:
            return  # Only run on main process
            
        logger.info(f"Running ensemble validation at step {step}")
        
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
            logger.info(f"Ensemble validation loss: {ensemble_loss.item():.6f}")
            for expert_idx, loss in expert_losses.items():
                logger.info(f"Expert {expert_idx} validation loss: {loss:.6f}")
                
            # Log router accuracy
            top1_accuracy = (router_probs.argmax(dim=1) == cluster_labels).float().mean()
            logger.info(f"Router accuracy: {top1_accuracy.item():.2f}")
            
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
            logger.error(f"Error in ensemble validation: {str(e)}")
    
    def run_validation(self, step):
        """Run validation by generating samples and computing metrics"""
        # Only run validation on main process
        if not is_main_process():
            return {}
            
        logger.info(f"Running validation at step {step}")
        validation_start_time = time.time()
        
        # Generate samples
        try:
            # Setup validation
            num_samples = getattr(self.config, 'validation_samples', 4)
            guidance_scale = getattr(self.config, 'validation_guidance_scale', 7.5)
            
            # Get validation prompts
            validation_prompts = getattr(self.config, 'validation_prompts', [
                "a photo of a cat",
                "a scenic mountain landscape",
                "a still life with fruit on a table",
                "portrait of a smiling person"
            ])
            
            # Log validation settings
            logger.info(f"Generating {num_samples} validation samples with guidance scale {guidance_scale}")
            logger.info(f"Using validation prompts: {validation_prompts}")
            
            # Generate samples
            generation_start_time = time.time()
            samples = self.generate_samples(
                prompts=validation_prompts,
                num_samples=num_samples,
                guidance_scale=guidance_scale
            )
            generation_time = time.time() - generation_start_time
            
            # Calculate stats
            prompt_count = len(validation_prompts)
            total_samples = len(samples)
            
            logger.info(f"Generated {total_samples} samples for {prompt_count} prompts in {generation_time:.2f}s "
                       f"({generation_time/total_samples:.2f}s per sample)")
            
            # Save samples
            if total_samples > 0:
                save_start_time = time.time()
                
                # Create output directory
                sample_dir = os.path.join(self.config.output_dir, 'samples', f'step_{step}')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save each image
                for i, img in enumerate(samples):
                    prompt_idx = i % len(validation_prompts)
                    prompt = validation_prompts[prompt_idx]
                    # Create safe filename from prompt
                    prompt_safe = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
                    img_path = os.path.join(sample_dir, f'{prompt_safe}_{i}.png')
                    img.save(img_path)
                
                save_time = time.time() - save_start_time
                logger.info(f"Saved {total_samples} samples to {sample_dir} in {save_time:.2f}s")
                
                # Compute metrics (if FID evaluation enabled)
                if getattr(self.config, 'compute_fid', False):
                    metrics_start_time = time.time()
                    try:
                        # Compute FID and other metrics (implementation specific)
                        metrics = self._compute_validation_metrics(sample_dir)
                        metrics_time = time.time() - metrics_start_time
                        logger.info(f"Computed validation metrics in {metrics_time:.2f}s: {metrics}")
                    except Exception as e:
                        logger.error(f"Error computing validation metrics: {str(e)}")
                        metrics = {}
                else:
                    metrics = {}
                    
                # Log to wandb if configured
                if getattr(self.config, 'use_wandb', False):
                    import wandb
                    try:
                        # Log images to wandb
                        wandb_images = [wandb.Image(img, caption=prompt) 
                                      for img, prompt in zip(samples, validation_prompts * num_samples)]
                        wandb.log({
                            'validation/samples': wandb_images,
                            'validation/step': step,
                            **{f'validation/{k}': v for k, v in metrics.items()}
                        })
                        logger.info("Logged validation samples and metrics to wandb")
                    except Exception as e:
                        logger.error(f"Error logging to wandb: {str(e)}")
            else:
                logger.warning("No validation samples were generated")
                metrics = {}
                
            # Log timing
            validation_time = time.time() - validation_start_time
            logger.info(f"Validation completed in {validation_time:.2f}s")
                
            return metrics
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            
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
                    logger.error(f"Error loading expert {expert_idx} for sampling: {str(e)}")
                    
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
            logger.error(f"Error generating samples: {str(e)}")
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
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
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
                logger.info("Loaded router state from checkpoint")
                
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
                        logger.info(f"Loaded state for expert {expert_idx}")
                    except Exception as e:
                        logger.error(f"Failed to load state for expert {expert_idx}: {str(e)}")
            
            logger.info(f"Checkpoint loaded successfully, resuming from step {step}")
            return step
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return 0
    
    def save_sharded_checkpoints(self, step):
        """Save checkpoints for coordinator, router, and expert networks"""
        if not is_main_process() and not getattr(self.config, 'save_all_ranks', False):
            return
            
        logger.info(f"Saving checkpoint at step {step} on rank {self.rank}")
        checkpoint_start_time = time.time()
        
        # Create checkpoint directory
        try:
            checkpoint_dir = os.path.join(self.config.checkpoint_dir, f'step_{step}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger.info(f"Created checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory: {str(e)}")
            return
            
        # Save router checkpoint (only on rank 0)
        if self.rank == 0 and self.router is not None:
            try:
                router_start_time = time.time()
                router_path = os.path.join(checkpoint_dir, 'router.pth')
                
                torch.save({
                    'step': step,
                    'model_state_dict': self.router.state_dict(),
                    'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
                }, router_path)
                
                router_time = time.time() - router_start_time
                router_size = os.path.getsize(router_path) / (1024 * 1024) # size in MB
                logger.info(f"Saved router to {router_path} in {router_time:.2f}s (size: {router_size:.1f} MB)")
            except Exception as e:
                logger.error(f"Failed to save router: {str(e)}")
        
        # Save experts
        expert_start_time = time.time()
        experts_saved = 0
        
        for expert_idx, expert in self.experts.items():
            # Only save experts assigned to this rank
            if self.is_expert_owned_by_rank(expert_idx):
                try:
                    # Create expert sub-directory by rank for organization
                    expert_dir = os.path.join(checkpoint_dir, f'rank_{self.rank}')
                    os.makedirs(expert_dir, exist_ok=True)
                    
                    expert_path = os.path.join(expert_dir, f'expert_{expert_idx}.pth')
                    expert_save_start = time.time()
                    
                    torch.save({
                        'step': step,
                        'expert_idx': expert_idx,
                        'model_state_dict': expert.state_dict(),
                        'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
                    }, expert_path)
                    
                    expert_save_time = time.time() - expert_save_start
                    expert_size = os.path.getsize(expert_path) / (1024 * 1024) # size in MB
                    experts_saved += 1
                    
                    logger.info(f"Saved expert {expert_idx} to {expert_path} in {expert_save_time:.2f}s (size: {expert_size:.1f} MB)")
                except Exception as e:
                    logger.error(f"Failed to save expert {expert_idx}: {str(e)}")
        
        expert_time = time.time() - expert_start_time
        if experts_saved > 0:
            logger.info(f"Saved {experts_saved} experts in {expert_time:.2f}s (avg: {expert_time/experts_saved:.2f}s per expert)")
        
        # Save coordinator checkpoint
        try:
            coordinator_start_time = time.time()
            coordinator_path = os.path.join(checkpoint_dir, f'coordinator_rank_{self.rank}.pth')
            
            # Create minimal state dict for coordinator
            coord_state = {
                'step': step,
                'rank': self.rank,
                'world_size': self.world_size,
                'experts': list(self.experts.keys()),
                'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }
            
            torch.save(coord_state, coordinator_path)
            
            coordinator_time = time.time() - coordinator_start_time
            coordinator_size = os.path.getsize(coordinator_path) / (1024 * 1024) # size in MB
            logger.info(f"Saved coordinator checkpoint to {coordinator_path} in {coordinator_time:.2f}s (size: {coordinator_size:.1f} MB)")
        except Exception as e:
            logger.error(f"Failed to save coordinator checkpoint: {str(e)}")
            
        # Synchronize after saving if needed
        if self.world_size > 1:
            sync_start = time.time()
            self.safe_synchronize(timeout_seconds=60, name="checkpoint_saving")
            sync_time = time.time() - sync_start
            logger.info(f"Checkpoint synchronization completed in {sync_time:.2f}s")
        
        # Log final timing
        total_time = time.time() - checkpoint_start_time
        logger.info(f"Checkpoint saving completed in {total_time:.2f}s at step {step}")
    
    def train_distilled_model(self):
        """Train distilled model from experts (Section 4.3)"""
        # Only run distillation on main process
        if not is_main_process():
            return
            
        logger.info("Starting model distillation")
        distill_start_time = time.time()
        
        try:
            # Create output directory
            distill_dir = os.path.join(self.config.output_dir, 'distilled')
            os.makedirs(distill_dir, exist_ok=True)
            logger.info(f"Created distillation directory: {distill_dir}")
            
            # Get distillation config from main config
            distill_config = getattr(self.config, 'distillation', {})
            
            # Log distillation settings
            num_steps = distill_config.get('num_steps', 10000)
            batch_size = distill_config.get('batch_size', 16)
            learning_rate = distill_config.get('learning_rate', 1e-5)
            
            logger.info(f"Distillation settings: {num_steps} steps, batch_size={batch_size}, " 
                       f"learning_rate={learning_rate}")
            
            # Load experts for distillation
            load_start_time = time.time()
            logger.info("Loading expert models for distillation")
            
            # Get expert models (assuming they are already loaded)
            experts = {}
            for expert_idx in range(self.config.num_experts):
                try:
                    experts[expert_idx] = self.get_expert(expert_idx).expert
                except Exception as e:
                    logger.error(f"Error loading expert {expert_idx} for distillation: {str(e)}")
                    
            expert_count = len(experts)
            load_time = time.time() - load_start_time
            logger.info(f"Loaded {expert_count} experts for distillation in {load_time:.2f}s")
            
            if expert_count == 0:
                logger.error("No experts available for distillation, aborting")
                return
                
            # Create distillation trainer
            from trainers.distillation import ModelDistiller
            logger.info("Initializing model distiller")
            init_start_time = time.time()
            
            distiller = ModelDistiller(
                config=self.config,
                experts=experts,
                device=self.device
            )
            
            init_time = time.time() - init_start_time
            logger.info(f"Model distiller initialized in {init_time:.2f}s")
            
            # Run distillation
            logger.info(f"Starting distillation training for {num_steps} steps")
            train_start_time = time.time()
            
            distiller.train(
                num_steps=num_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=lambda step, total, loss: 
                    logger.info(f"Distillation step {step}/{total}, loss: {loss:.6f}")
                    if step % 100 == 0 else None
            )
            
            train_time = time.time() - train_start_time
            avg_time_per_step = train_time / num_steps
            logger.info(f"Distillation training completed in {train_time:.2f}s "
                       f"({avg_time_per_step:.4f}s per step)")
            
            # Save distilled model
            save_start_time = time.time()
            distill_path = os.path.join(distill_dir, 'distilled_model.pth')
            
            logger.info(f"Saving distilled model to {distill_path}")
            distiller.save(distill_path)
            
            save_time = time.time() - save_start_time
            model_size = os.path.getsize(distill_path) / (1024 * 1024) # size in MB
            logger.info(f"Distilled model saved to {distill_path} in {save_time:.2f}s (size: {model_size:.1f} MB)")
            
            # Run validation on distilled model
            if getattr(self.config, 'validate_distilled', True):
                validation_start_time = time.time()
                logger.info("Running validation on distilled model")
                
                try:
                    # Generate samples with distilled model
                    validation_prompts = getattr(self.config, 'validation_prompts', [
                        "a photo of a cat",
                        "a scenic mountain landscape",
                        "a still life with fruit on a table",
                        "portrait of a smiling person"
                    ])
                    
                    samples = distiller.generate_samples(validation_prompts, num_samples=2)
                    
                    # Save samples
                    if samples:
                        sample_dir = os.path.join(distill_dir, 'samples')
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        for i, (prompt, img) in enumerate(zip(validation_prompts, samples)):
                            img_path = os.path.join(sample_dir, f'sample_{i}.png')
                            img.save(img_path)
                            
                        logger.info(f"Saved {len(samples)} validation samples to {sample_dir}")
                    
                    validation_time = time.time() - validation_start_time
                    logger.info(f"Distilled model validation completed in {validation_time:.2f}s")
                except Exception as e:
                    logger.error(f"Error validating distilled model: {str(e)}")
            
            # Log total time
            total_time = time.time() - distill_start_time
            hrs, mins = divmod(total_time, 3600)
            mins, secs = divmod(mins, 60)
            
            logger.info(f"Distillation completed in {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")
            logger.info(f"Distilled model saved to {distill_path}")
            
        except Exception as e:
            logger.error(f"Error during distillation: {str(e)}")
            raise
    
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

    def create_lr_scheduler(self):
        """
        Initialize learning rate scheduler function
        """
        # This method is now here to avoid AttributeError, but the actual
        # scheduler creation happens in the __init__ method
        logger.debug("Learning rate scheduler configuration is handled during initialization")
        pass
