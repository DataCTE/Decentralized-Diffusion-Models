"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""


import math
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tqdm import tqdm
import numpy as np
import wandb
from torch.distributed.fsdp import FullStateDictConfig, StateDictType


from models.dit import ExpertDiT
from models.router import RouterModel
from data.dataset import DDMDataset, FeatureDataset
from utils.diffusion import DecentralizedFlowMatcher
from utils.fsdp import create_fsdp_config
from data.clustering import ClusterManager

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
        """Initialize models using dedicated trainer classes"""
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
        
    def init_optimizers(self):
        """Initialize optimizers following paper Section 4.1"""
        self.expert_optims = [torch.optim.AdamW8bit(expert.parameters(), 
                                              lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)
                            for expert in self.expert_trainers]
        
        self.router_optim = torch.optim.AdamW8bit(self.router_trainer.router.parameters(),
                                            lr=self.config.router_learning_rate,
                                            weight_decay=self.config.weight_decay)
        
    def run_training_cycle(self):
        """Paper Algorithm 2: Main training loop with full sharding"""
        logger.info("Starting DDM training with dedicated trainers...")
        
        # Paper-recommended training schedule
        for step in range(self.config.num_steps):
            # Expert training phase (Section 3.2)
            expert_losses = self.train_experts(step)
            
            # Router training phase (Section 3.3)
            router_loss = self.train_router(step)
            
            # Paper-mandated synchronization points
            if self.needs_reclustering(step):
                self.perform_reclustering()
                self.run_validation(step)
                
            # Distributed metric logging
            self.log_sharded_metrics(step, expert_losses, router_loss)

    def train_experts(self, step):
        """Delegate expert training to ExpertTrainer instances"""
        return np.mean([
            trainer.train_step(batch)
            for trainer, loader in zip(self.expert_trainers, self.expert_loaders)
            for batch in loader
        ])
    
    def train_router(self, step):
        """Delegate router training to RouterTrainer"""
        return self.router_trainer.train_epoch(self.router_loader)
    
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
        """Paper-aligned validation with sharded models"""
        if self.rank != 0: 
            return
        
        # Gather sharded model states
        expert_models = []
        with FSDP.summon_full_params(self.expert_trainers[0].expert):
            for trainer in self.expert_trainers:
                expert_models.append(trainer.expert.clone().to("cpu"))
        
        # Generate samples using paper's algorithm
        samples = DecentralizedFlowMatcher.sample(
            self.config,
            self.router_trainer.router,
            expert_models,
            num_samples=4,
            device="cpu"
        )
        
        # Calculate metrics
        fid = self.calculate_fid(samples, self.val_dataset)
        self.log_to_wandb(step, fid, samples)
        
    def save_sharded_checkpoints(self, step):
        """Save checkpoints through trainer classes"""
        if self.rank == 0:
            logger.info(f"Saving sharded checkpoint at step {step}")
        
        checkpoint = {
            'experts': [trainer.expert.state_dict() for trainer in self.expert_trainers],
            'router': self.router_trainer.router.state_dict(),
            'config': self.config,
            'step': step
        }
        
        torch.save(checkpoint, f"{self.config.save_dir}/ddm_step{step}.pt")
            
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
        """Paper Section 4.3: Frechet Inception Distance calculation"""
        # Use paper-recommended InceptionV3 features
        fid_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        fid_model.eval().to(self.device)
        fid_model.fc = torch.nn.Identity()  # Remove final layer
        
        def extract_features(dataloader):
            features = []
            for batch in tqdm(dataloader, desc="Extracting FID features"):
                images = batch['image'].to(self.device)
                with torch.no_grad():
                    feats = fid_model(images)
                features.append(feats.cpu())
            return torch.cat(features)
        
        # Real data features
        real_loader = DataLoader(real_dataset, batch_size=256, num_workers=4)
        real_features = extract_features(real_loader)
        
        # Generated samples features
        gen_loader = DataLoader(generated_samples, batch_size=256, num_workers=4)
        gen_features = extract_features(gen_loader)
        
        # Compute FID
        mu_real, sigma_real = real_features.mean(dim=0), torch.cov(real_features.T)
        mu_gen, sigma_gen = gen_features.mean(dim=0), torch.cov(gen_features.T)
        
        diff = mu_real - mu_gen
        cov_mean = (sigma_real @ sigma_gen).sqrt()
        fid = diff.dot(diff) + torch.trace(sigma_real + sigma_gen - 2*cov_mean)
        return fid.item()

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
