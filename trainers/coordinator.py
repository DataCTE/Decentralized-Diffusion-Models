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


from models.dit import ExpertDiT
from data.dataset import DDMDataset, FeatureDataset
from utils.diffusion import DecentralizedFlowMatcher
from data.clustering import ClusterManager
from trainers.Distillation import DiffusionDistiller
from utils.clip import CLIPTextEncoder

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
        
        self.loaded_experts = {}  # Track loaded experts
        self.expert_loading_count = 0
        
    def init_cluster_manager(self):
        """Initialize clustering components as per paper Section 3.2"""
        self.cluster_manager = ClusterManager()
        self.previous_clusters = None
        
        # Perform initial clustering if not already done
        if self.cluster_manager.get_clusters() is None:
            feature_loader = DataLoader(
                FeatureDataset(self.config.dataset_path, self.config),
                batch_size=self.config.feature_batch_size,
                num_workers=self.config.feature_workers
            )
            self.cluster_manager.perform_clustering(feature_loader)

    def init_data_loaders(self):
        """Initialize distributed data loaders with sharded validation"""
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

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )
        
    def perform_initial_clustering(self):
        """Paper Algorithm 1: Initial dataset clustering"""
        # First stage: 1024 fine-grained clusters
        feature_dataset = FeatureDataset(self.config.dataset_path, self.config)
        feature_loader = DataLoader(feature_dataset, 
                                  batch_size=self.config.feature_batch_size,
                                  num_workers=self.config.feature_workers)
        fine_features = self.cluster_manager.extract_features(feature_loader)
        fine_clusters = self.cluster_manager.cluster_dataset(fine_features, n_clusters=1024)

        # Second stage: Consolidate to coarse clusters
        coarse_clusters = self.cluster_manager.consolidate_clusters(
            fine_clusters, target_clusters=self.config.num_experts
        )
        return coarse_clusters
    
    def sync_cluster_labels(self):
        """Synchronize cluster labels across all processes"""
        cluster_tensor = torch.from_numpy(self.cluster_labels).to(self.device)
        dist.broadcast(cluster_tensor, src=0)
        self.cluster_labels = cluster_tensor.cpu().numpy()
        
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
        """Initialize optimizers following paper Section 4.1"""
        self.expert_optims = [torch.optim.AdamW8bit(expert.parameters(), 
                                              lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)
                            for expert in self.expert_trainers]
        
        self.router_optim = torch.optim.AdamW8bit(self.router_trainer.router.parameters(),
                                            lr=self.config.router_learning_rate,
                                            weight_decay=self.config.weight_decay)
        
    def train_experts(self, step):
        """Delegate expert training to ExpertTrainer instances"""
        loss = np.mean([
            trainer.train_step(batch)
            for trainer, loader in zip(self.expert_trainers, self.expert_loaders)
            for batch in loader
        ])
        
        # Add diversity loss
        diversity_loss = torch.stack([e.diversity() for e in self.expert_trainers]).mean()
        loss += self.config.diversity_lambda * diversity_loss
        
        return loss
    
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
        
        # New expert data migration
        cluster_mapping = self.cluster_manager.get_cluster_mapping(
            self.previous_clusters, new_clusters
        )
        
        # Migrate expert parameters
        for old_idx, new_idx in cluster_mapping.items():
            if old_idx != new_idx:
                self.migrate_expert_data(old_idx, new_idx)
        
        # Update validation dataset assignments
        self.val_dataset.update_clusters(new_clusters)
        
        # Paper-recommended expert reset for empty clusters
        for new_idx in range(self.config.num_experts):
            if new_idx not in cluster_mapping.values():
                self.expert_trainers[new_idx].reset_parameters()
        
        # Update previous clusters
        self.previous_clusters = new_clusters
        
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
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        checkpoint = {
            'meta': {'step': step, 'best_fid': self.best_fid},
            'router': FSDP.state_dict(self.router_trainer.router),
        }
        
        # Save experts separately using FSDP sharded format
        expert_paths = []
        for idx, trainer in enumerate(self.expert_trainers):
            expert_state = FSDP.state_dict(trainer.expert)
            path = f"{self.config.save_dir}/expert_{idx}_step{step}_v{self.config.version}.pt"
            if self.rank == 0:
                torch.save(expert_state, path)
            expert_paths.append(path)
        
        checkpoint['expert_paths'] = expert_paths
        
        if self.rank == 0:
            torch.save(checkpoint, f"{self.config.save_dir}/coordinator_step{step}.pt")
            
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
        fid_inception = fid.item()
        
        # New CLIP-FID calculation
        clip_model = CLIPTextEncoder(self.device, self.config).model.visual
        clip_model.eval()
        
        def clip_features(images):
            with torch.no_grad():
                return clip_model(images)
        
        # DINOv2-FID calculation
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dino_model.eval().to(self.device)
        
        def dino_features(images):
            with torch.no_grad():
                return dino_model(images)
        
        # Compute all metrics
        return {
            'inception_fid': fid_inception,
            'clip_fid': self._compute_fid_pair(clip_features, real_loader, gen_loader),
            'dino_fid': self._compute_fid_pair(dino_features, real_loader, gen_loader)
        }

    def _compute_fid_pair(self, feature_fn, real_loader, gen_loader):
        """Helper to compute FID for different feature extractors"""
        real_feats = torch.cat([feature_fn(batch['image'].to(self.device)) 
                              for batch in real_loader])
        gen_feats = torch.cat([feature_fn(batch.to(self.device)) 
                             for batch in gen_loader])
        
        mu_real, sigma_real = torch.mean(real_feats, 0), torch.cov(real_feats.T)
        mu_gen, sigma_gen = torch.mean(gen_feats, 0), torch.cov(gen_feats.T)
        
        return ((mu_real - mu_gen).pow(2).sum() + 
                torch.trace(sigma_real + sigma_gen - 2*(sigma_real@sigma_gen).sqrt())).item()

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
