"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""

import os
import math
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.cluster import MiniBatchKMeans


from models.dit import ExpertDiT
from models.router import RouterModel
from utils.dataset import DDMDataset, FeatureDataset
from utils.diffusion import DecentralizedFlowMatcher

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
        
        # Paper-recommended initialization sequence
        self.init_cluster_manager()
        self.init_data_loaders()
        self.init_models()
        self.init_optimizers()
        
        # Metrics tracking (paper Section 4.3)
        self.best_fid = float('inf')
        self.train_losses = []
        
    def init_cluster_manager(self):
        """Initialize clustering components as per paper Section 3.2"""
        self.cluster_manager = ClusterManager()
        self.previous_clusters = None

    def init_data_loaders(self):
        """Initialize distributed data loaders following paper Appendix A.1"""
        # Only rank 0 handles dataset initialization
        if self.rank == 0:
            self.full_dataset = DDMDataset(self.config.dataset_path)
            self.cluster_labels = self.perform_initial_clustering()
        else:
            self.full_dataset = DDMDataset(self.config.dataset_path)
            self.cluster_labels = np.zeros(len(self.full_dataset), dtype=np.int32)
            
        # Sync cluster labels across all nodes
        self.sync_cluster_labels()
        
        # Create expert-specific loaders (paper Section 3.3)
        self.expert_loaders = self.create_expert_loaders()
        
    def perform_initial_clustering(self):
        """Paper Algorithm 1: Initial dataset clustering"""
        feature_dataset = FeatureDataset(self.config.dataset_path, self.config)
        feature_loader = DataLoader(feature_dataset, 
                                  batch_size=self.config.feature_batch_size,
                                  num_workers=self.config.feature_workers)
        
        features = self.cluster_manager.extract_features(feature_loader)
        return self.cluster_manager.cluster_dataset(features)
    
    def sync_cluster_labels(self):
        """Synchronize cluster labels across all processes"""
        cluster_tensor = torch.from_numpy(self.cluster_labels).to(self.device)
        dist.broadcast(cluster_tensor, src=0)
        self.cluster_labels = cluster_tensor.cpu().numpy()
        
    def init_models(self):
        """Initialize models with FSDP as per paper Appendix A.2"""
        # Expert models
        self.experts = [
            FSDP(ExpertDiT(self.config), 
                 device_id=self.device,
                 sharding_strategy=ShardingStrategy.FULL_SHARD)
            for _ in range(self.config.num_experts)
        ]
        
        # Router model
        self.router = FSDP(RouterModel(self.config),
                          device_id=self.device,
                          sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
        
        # Paper-recommended initialization
        if self.rank == 0:
            logger.info(f"Initialized {self.config.num_experts} experts and router with FSDP")
            
    def init_optimizers(self):
        """Initialize optimizers following paper Section 4.1"""
        self.expert_optims = [torch.optim.AdamW(expert.parameters(), 
                                              lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)
                            for expert in self.experts]
        
        self.router_optim = torch.optim.AdamW(self.router.parameters(),
                                            lr=self.config.router_learning_rate,
                                            weight_decay=self.config.weight_decay)
        
    def run_training_cycle(self):
        """Paper Algorithm 2: Main training loop"""
        logger.info("Starting DDM training...")
        
        # Paper-recommended training schedule
        for step in range(self.config.num_steps):
            # Expert training phase
            expert_losses = self.train_experts(step)
            
            # Router training phase
            router_loss = self.train_router(step)
            
            # Reclustering and validation
            if self.needs_reclustering(step):
                self.perform_reclustering()
                self.run_validation(step)
                
            # Checkpointing
            if step % self.config.save_interval == 0:
                self.save_checkpoints(step)
                
            # Logging
            self.log_metrics(step, expert_losses, router_loss)
            
    def train_experts(self, step):
        """Paper Section 3.2: Expert training with flow matching"""
        losses = []
        for expert_idx, expert in enumerate(self.experts):
            for batch in self.expert_loaders[expert_idx]:
                # Forward pass
                loss = self.flow_matching_loss(expert, batch)
                
                # Backward pass
                self.expert_optims[expert_idx].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expert.parameters(), 
                                             self.config.max_grad_norm)
                self.expert_optims[expert_idx].step()
                
                losses.append(loss.item())
                
        return np.mean(losses)
    
    def train_router(self, step):
        """Paper Section 3.3: Router training"""
        losses = []
        for batch in self.router_loader:
            # Get router predictions
            logits = self.router(batch['latent'], batch['t'])
            
            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(logits, batch['cluster'])
            
            # Optimization
            self.router_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(),
                                         self.config.max_grad_norm)
            self.router_optim.step()
            
            losses.append(loss.item())
            
        return np.mean(losses)
    
    def flow_matching_loss(self, expert, batch):
        """Paper Equation 6: Flow matching objective"""
        # Forward process
        t = torch.rand(batch['image'].size(0), device=self.device)
        noise = torch.randn_like(batch['latent'])
        alpha_t = torch.cos(t * math.pi/2)[:, None, None, None]
        sigma_t = torch.sin(t * math.pi/2)[:, None, None, None]
        x_t = alpha_t * batch['latent'] + sigma_t * noise
        
        # Expert prediction
        pred_flow = expert(x_t, t, batch['text_embeds'])
        
        # Target flow
        target_flow = (batch['latent'] - x_t) / sigma_t
        
        return torch.nn.functional.mse_loss(pred_flow, target_flow)
    
    def perform_reclustering(self):
        """Paper Section 3.3: Dynamic reclustering"""
        logger.info("Performing dynamic reclustering...")
        
        # Extract features from current dataset
        features = self.cluster_manager.extract_features(
            DataLoader(FeatureDataset(self.config.dataset_path, self.config),
                     batch_size=self.config.feature_batch_size)
        )
        
        # Update clusters
        new_clusters = self.cluster_manager.cluster_dataset(features)
        self.update_cluster_assignments(new_clusters)
        
    def run_validation(self, step):
        """Paper Section 4.3: Validation and metrics"""
        # Generate samples
        samples = self.ddm_sample(num_samples=4)
        
        # Calculate FID
        fid_score = self.calculate_fid(samples)
        
        # Log results
        if self.rank == 0:
            self.log_to_wandb(step, fid_score, samples)
            
    def save_checkpoints(self, step):
        """Save model checkpoints with FSDP state dicts"""
        checkpoint = {
            'experts': [expert.state_dict() for expert in self.experts],
            'router': self.router.state_dict(),
            'config': self.config,
            'step': step
        }
        
        if self.rank == 0:
            torch.save(checkpoint, 
                      f"{self.config.save_dir}/ddm_step{step}.pt")
            logger.info(f"Saved checkpoint at step {step}")
            
    def log_metrics(self, step, expert_loss, router_loss):
        """Log training metrics to WandB"""
        if self.rank == 0:
            metrics = {
                'expert_loss': expert_loss,
                'router_loss': router_loss,
                'learning_rate': self.get_current_lr(),
                'step': step
            }
            wandb.log(metrics)
            
    # Helper methods omitted for brevity (ddm_sample, calculate_fid, get_current_lr, etc.)
    
class ClusterManager:
    """Implements paper's clustering strategy from Section 3.2"""
    def __init__(self):
        self.feature_extractor = DINOv2FeatureExtractor()
        self.kmeans = MiniBatchKMeans(n_clusters=1024)
        
    def extract_features(self, dataloader):
        """Extract DINOv2 features from dataset"""
        features = []
        for batch in dataloader:
            features.append(self.feature_extractor(batch))
        return torch.cat(features).cpu().numpy()
    
    def cluster_dataset(self, features):
        """Two-stage clustering from paper Appendix A.1"""
        # Stage 1: Fine-grained clustering
        self.kmeans.fit(features)
        fine_labels = self.kmeans.labels_
        
        # Stage 2: Coarse clustering
        centroids = self.kmeans.cluster_centers_
        coarse_kmeans = MiniBatchKMeans(n_clusters=self.config.num_experts)
        coarse_kmeans.fit(centroids)
        
        return coarse_kmeans.labels_[fine_labels]
