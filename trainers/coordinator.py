"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""


import math
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import wandb
import time
import os

from models.dit import ExpertDiT
from data.dataset import DDMDataset, FeatureDataset, create_expert_bucket_loaders
from utils.diffusion import DecentralizedFlowMatcher
from data.clustering import ClusterManager
from trainers.Distillation import DiffusionDistiller
from utils.clip import CLIPTextEncoder
from utils.checkpoint import save_sharded
from utils.metrics import MetricCalculator

logger = logging.getLogger(__name__)

class DDMTrainingCoordinator:
    """Implements the core training logic from Section 3 of the paper"""
    
    def __init__(self, config, rank, world_size):
        """
        Initialize coordinator as per paper Section 4.1
        Args:
            config: Configuration object
            rank: Process rank (0 is main)
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.current_step = 0
        
        if self.rank == 0:
            logger.info(f"Initializing DDM Training Coordinator with {world_size} processes")
            logger.info(f"Training will use {world_size} GPUs in parallel")
            logger.info(f"Beginning initialization sequence - this may take 20-30 minutes")
        
        # Paper-recommended initialization sequence with better logging
        start_time = time.time()
        if self.rank == 0:
            logger.info("Stage 1: Initializing cluster manager")
        self.init_cluster_manager()
        
        if self.rank == 0:
            clustering_time = time.time() - start_time
            logger.info(f"Clustering completed in {clustering_time/60:.1f} minutes")
            logger.info("Stage 2: Initializing data loaders")
            
        loader_start = time.time()
        self.init_data_loaders()
        
        if self.rank == 0:
            loader_time = time.time() - loader_start
            logger.info(f"Data loaders initialized in {loader_time:.1f} seconds")
            logger.info("Stage 3: Initializing models")
            
        model_start = time.time()
        self.init_models()
        
        if self.rank == 0:
            model_time = time.time() - model_start
            logger.info(f"Models initialized in {model_time:.1f} seconds")
            logger.info("Stage 4: Initializing optimizers")
            
        opt_start = time.time()
        self.init_optimizers()
        
        if self.rank == 0:
            opt_time = time.time() - opt_start
            total_time = time.time() - start_time
            logger.info(f"Optimizers initialized in {opt_time:.1f} seconds")
            logger.info(f"Total initialization time: {total_time/60:.1f} minutes")
            logger.info("Initialization complete - ready to begin training")
        
        # Metrics tracking (paper Section 4.3)
        self.best_fid = float('inf')
        self.train_losses = []
        
        self.loaded_experts = {}  # Track loaded experts
        self.expert_loading_count = 0
        
    def init_cluster_manager(self):
        """Initialize clustering components with distributed synchronization"""
        # Log initialization start
        if self.rank == 0:
            logger.info(f"Initializing clustering manager (rank {self.rank})")
            logger.info(f"This process involves feature extraction and clustering")
            logger.info(f"GPUs will be at 100% during feature extraction - this is normal")
            logger.info(f"Process steps: Extract DINOv2 features → Fine clustering → Coarse clustering")
        
        # Create cluster manager with dataset path and config
        self.cluster_manager = ClusterManager(
            local_rank=self.rank,
            dataset_path=self.config.dataset_path,
            config=self.config
        )
        
        # Centralized clustering call - this handles everything including dataloader creation
        self.cluster_manager.perform_clustering()
        
        if self.rank == 0:
            logger.info(f"Clustering phase complete - all processes synchronized")

    def init_data_loaders(self):
        """Initialize distributed data loaders with sharded validation"""
        # Log initialization start
        if self.rank == 0:
            logger.info(f"Initializing data loaders with distributed sharding")
            logger.info(f"Each GPU will process a different subset of the data")
            
        start_time = time.time()
            
        # Shared dataset for all processes
        dataset = DDMDataset(
            self.config.dataset_path,
            cluster_labels=self.cluster_manager.get_clusters()
        )

        # Distributed sampler for sharding
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )

        if self.rank == 0:
            logger.info(f"Creating training dataloader with batch size {self.config.batch_size}")
            
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
        if self.rank == 0:
            setup_time = time.time() - start_time
            logger.info(f"Data loaders initialized in {setup_time:.2f} seconds")
            logger.info(f"Each process will handle ~{len(dataset)/self.world_size:,.0f} images")
    
    def perform_reclustering(self):
        """Perform dynamic reclustering during training using ClusterManager"""
        if self.rank == 0:
            logger.info("Starting dynamic reclustering process")
            
        # Let ClusterManager handle all reclustering logic
        old_clusters, new_clusters = self.cluster_manager.perform_reclustering()
        
        # Get mapping from old to new clusters
        cluster_mapping = self.cluster_manager.get_cluster_mapping(old_clusters, new_clusters)
        
        # Migrate expert parameters based on mapping
        if self.rank == 0:
            logger.info("Migrating expert parameters based on new clustering")
            
        for old_idx, new_idx in cluster_mapping.items():
            if old_idx != new_idx:
                if self.rank == 0:
                    logger.info(f"Migrating expert data from {old_idx} to {new_idx}")
                self.migrate_expert_data(old_idx, new_idx)
        
        # Update validation dataset assignments
        if hasattr(self, 'val_dataset'):
            if self.rank == 0:
                logger.info("Updating validation dataset cluster assignments")
            self.val_dataset.update_clusters(new_clusters)
        
        # Paper-recommended expert reset for empty clusters
        for expert_idx in range(self.config.num_experts):
            if expert_idx not in cluster_mapping.values():
                if self.rank == 0:
                    logger.info(f"Resetting parameters for empty expert {expert_idx}")
                self.expert_trainers[expert_idx].reset_parameters()
        
        if self.rank == 0:
            logger.info("Dynamic reclustering complete")
        
        return new_clusters
    
    def init_models(self):
        """Initialize models with parameter isolation checks"""
        from trainers.expert import ExpertTrainer
        from trainers.router import RouterTrainer
        
        # Initialize expert trainers
        self.expert_trainers = [
            ExpertTrainer(
                expert_idx=i,
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            ) for i in range(self.config.num_experts)
        ]
        
        # Initialize router trainer
        self.router_trainer = RouterTrainer(
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
        
        if self.rank == 0:
            logger.info(f"Initialized {len(self.expert_trainers)} experts and router with dedicated trainers")
        
        expert_params = set()
        for i in range(self.config.num_experts):
            expert = ExpertDiT(self.config)
            current_params = {id(p) for p in expert.parameters()}
            if expert_params & current_params:
                raise RuntimeError(f"Parameter sharing detected in expert {i}")
            expert_params |= current_params
        
    def init_optimizers(self):
        """Initialize optimizers - REMOVED implementation as optimizers are already initialized in respective trainers"""
        # The optimizers are already initialized in the respective trainers,
        # so we don't need to create duplicate optimizers here.
        # Just log that we're using the trainers' optimizers
        if self.rank == 0:
            logger.info(f"Using optimizers from {len(self.expert_trainers)} expert trainers and router trainer")
        
    def train_experts(self, step):
        """Delegate expert training to ExpertTrainer instances"""
        # Since each ExpertTrainer handles its own optimization,
        # we just need to call train_step on each trainer
        losses = []
        for trainer, loader in zip(self.expert_trainers, self.expert_loaders):
            for batch in loader:
                loss = trainer.train_step(batch)
                losses.append(loss)
            
        # Calculate mean loss
        mean_loss = np.mean(losses) if losses else 0.0
        
        # Add diversity loss if needed
        if hasattr(self.expert_trainers[0], 'diversity') and hasattr(self.config, 'diversity_lambda'):
            diversity_loss = torch.stack([e.diversity() for e in self.expert_trainers]).mean()
            mean_loss += self.config.diversity_lambda * diversity_loss.item()
        
        return mean_loss
    
    def train_router(self, step):
        """Delegate router training to RouterTrainer"""
        # The RouterTrainer handles its own optimization logic internally,
        # so we just need to call train_epoch and get the resulting loss
        loss = self.router_trainer.train_epoch(self.router_loader)
        
        # Update learning rate scheduler if present
        if hasattr(self.router_trainer, 'lr_scheduler'):
            self.router_trainer.lr_scheduler.step()
        
        return loss
    
    def run_validation(self, step):
        """Modified validation with confidence checks"""
        fallback_count = 0
        total_samples = 0
        
        samples = []
        for _ in range(self.config.validation_samples):
            with torch.no_grad():
                latent = torch.randn(...)
                for t in reversed(range(0, 1000)):
                    probs = self.router_trainer.router(latent, t)
                    max_prob = probs.max(dim=-1)[0]
                    
                    # Apply confidence threshold
                    if self.config.router_confidence_threshold > 0:
                        low_conf = max_prob < self.config.router_confidence_threshold
                        if low_conf.any():
                            # Use fallback expert for low confidence samples
                            probs[low_conf] = 0
                            probs[low_conf, 0] = 1.0  # Assign to expert 0
                            fallback_count += low_conf.sum().item()
                            total_samples += low_conf.size(0)
                    
                    top_k = torch.topk(probs, self.config.validation_topk)
                    
                    # Aggregate only needed experts
                    combined_flow = 0
                    for expert_idx in top_k.indices:
                        expert = self.get_expert(expert_idx.item())
                        flow = expert(latent, t)
                        combined_flow += flow * top_k.values[expert_idx]
                    
                    # Update latent
                    latent -= combined_flow * self.config.step_size
            
            samples.append(self.vae.decode(latent))
        
        # Calculate metrics
        fid = self.calculate_fid(samples, self.val_dataset)
        self.log_to_wandb(step, fid, samples)
        
        # Calibrate router after validation
        if step % self.config.calibration_interval == 0:
            self.router_trainer.calibrate_confidence(self.val_loader)
        
        # Log fallback usage
        metrics = {
            'fallback_rate': fallback_count / total_samples if total_samples > 0 else 0,
            'fid': fid,
            'best_fid': min(fid, self.best_fid),
            'samples': [wandb.Image(sample) for sample in samples]
        }
        wandb.log(metrics, step=step)
        
        # Update best FID
        if fid < self.best_fid:
            self.best_fid = fid
            torch.save({
                'experts': [trainer.expert.state_dict() for trainer in self.expert_trainers],
                'router': self.router_trainer.router.state_dict()
            }, f"{self.config.save_dir}/best_fid.pt")

    def save_sharded_checkpoints(self, step):
        """Paper-recommended sharded checkpoint format"""
        # Use utils.checkpoint to save coordinator metadata
        from utils.checkpoint import save_sharded
        
        # Metadata for the coordinator checkpoint
        meta_checkpoint = {
            'meta': {
                'step': step, 
                'best_fid': self.best_fid,
                'config': self.config.__dict__
            }
        }
        
        # Have experts save themselves through their own methods
        expert_paths = []
        for idx, trainer in enumerate(self.expert_trainers):
            # Check if trainer has save_checkpoint method, otherwise use fallback
            if hasattr(trainer, 'save_checkpoint'):
                path = trainer.save_checkpoint(self.config.save_dir, step)
                if path:  # Save path might be None for non-rank-0 processes
                    expert_paths.append(path)
            else:
                # Fallback for trainers without save_checkpoint method
                if self.rank == 0:
                    logger.warning(f"Expert trainer {idx} has no save_checkpoint method, using fallback")
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                
                path = f"{self.config.save_dir}/expert_{idx}_step{step}_v{self.config.version}.pt"
                if isinstance(trainer.expert, FSDP) and self.rank == 0:
                    state_dict = FSDP.state_dict(trainer.expert)
                    torch.save(state_dict, path)
                    expert_paths.append(path)
        
        # Save router model
        if self.rank == 0:
            if hasattr(self.router_trainer, 'save_checkpoint'):
                router_path = self.router_trainer.save_checkpoint(self.config.save_dir, step)
                meta_checkpoint['router_path'] = router_path
            else:
                # Fallback for router without save_checkpoint method
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                
                router_path = f"{self.config.save_dir}/router_step{step}.pt"
                if isinstance(self.router_trainer.router, FSDP):
                    state_dict = FSDP.state_dict(self.router_trainer.router)
                    torch.save(state_dict, router_path)
                else:
                    torch.save(self.router_trainer.router.state_dict(), router_path)
                meta_checkpoint['router_path'] = router_path
        
        # Save paths to expert checkpoints
        meta_checkpoint['expert_paths'] = expert_paths
        
        # Save coordinator checkpoint with metadata
        if self.rank == 0:
            os.makedirs(self.config.save_dir, exist_ok=True)
            coord_path = f"{self.config.save_dir}/coordinator_step{step}.pt"
            torch.save(meta_checkpoint, coord_path)
            logger.info(f"Saved coordinator checkpoint to {coord_path} with {len(expert_paths)} expert paths")
            
    def log_sharded_metrics(self, step, expert_loss, router_loss):
        """Log training metrics to WandB"""
        if self.rank == 0:
            metrics = {
                'expert_loss': expert_loss,
                'router_loss': router_loss,
                'learning_rate': self.get_current_lr(),
                'step': step
            }
            wandb.log(metrics)
            
    def calculate_fid(self, generated_samples, real_dataset):
        """Calculate FID score between generated and real samples"""
        
        # Use the centralized MetricCalculator instead of duplicating the logic
        return MetricCalculator.fid(real_dataset, generated_samples)
        
    def get_current_lr(self):
        """Paper-recommended learning rate schedule"""
        # Linear warmup over first 5% of training
        warmup_steps = int(0.05 * self.config.num_steps)
        if self.current_step < warmup_steps:
            return self.config.learning_rate * (self.current_step / warmup_steps)
        
        # Cosine decay for remaining steps
        progress = (self.current_step - warmup_steps) / (self.config.num_steps - warmup_steps)
        return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def log_to_wandb(self, step, fid, samples):
        """Paper Section 4.3: Logging metrics and samples"""
        if self.rank == 0:
            # Log metrics
            metrics = {
                'fid': fid,
                'best_fid': min(fid, self.best_fid),
                'samples': [wandb.Image(sample) for sample in samples]
            }
            wandb.log(metrics, step=step)
            
            # Update best FID
            if fid < self.best_fid:
                self.best_fid = fid
                torch.save({
                    'experts': [trainer.expert.state_dict() for trainer in self.expert_trainers],
                    'router': self.router_trainer.router.state_dict()
                }, f"{self.config.save_dir}/best_fid.pt")

    def train_distilled_model(self):
        """Paper Section 3.6: Knowledge Distillation"""
        teacher = DecentralizedFlowMatcher(
            self.config, self.router_trainer.router, self.expert_trainers
        )
        student = ExpertDiT(self.config).to(self.device)
        
        distiller = DiffusionDistiller(
            teacher=teacher,
            student=student,
            num_train_timesteps=self.config.num_timesteps,
            loss_fn=nn.MSELoss(),
            lr=self.config.distill_lr,
            warmup_ratio=0.05
        )
        
        # Train and save distilled model
        distiller.train(self.distill_loader)
        if self.rank == 0:
            torch.save(student.state_dict(), f"{self.config.save_dir}/distilled_model.pt")

    def needs_reclustering(self, step):
        """Paper's dynamic reclustering schedule"""
        return step % self.config.recluster_interval == 0

    def get_expert(self, idx):
        """Lazy-load experts with fallback mechanism"""
        try:
            return super().get_expert(idx)
        except KeyError:
            if self.config.router_confidence_threshold > 0:
                logger.warning(f"Using fallback expert for {idx}")
                return self.expert_trainers[0]
            raise

    def migrate_expert_data(self, old_idx, new_idx):
        """Transfer expert parameters and training data"""
        # Parameter transfer
        self.expert_trainers[new_idx].load_state_dict(
            self.expert_trainers[old_idx].state_dict()
        )
        
        # Data migration
        old_mask = (self.cluster_labels == old_idx)
        self.cluster_labels[old_mask] = new_idx
        
        # Update data loaders
        self.expert_loaders[new_idx].dataset.add_samples(
            self.expert_loaders[old_idx].dataset.remove_samples(old_mask)
        )
