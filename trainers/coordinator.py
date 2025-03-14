"""Training coordinator for Decentralized Diffusion Models (Paper implementation)"""

import math
import torch
import torch.nn as nn
import numpy as np
import time
import os

from models.dit import ExpertDiT
from data.dataset import DDMDataset
from data.clustering import ClusterManager
from trainers.Distillation import DiffusionDistiller
from trainers.expert import ExpertTrainer
from trainers.router import RouterTrainer

# Import centralized utilities
from utils.logging import setup_logger, log_metrics, log_images, log_training_start, log_training_end, init_wandb
from utils.distributed import (
    is_main_process, get_rank, get_world_size, synchronize, 
    broadcast_object, reduce_dict, all_gather_tensor
)
from utils.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from utils.metrics import MetricCalculator
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint, save_sharded
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder
from utils.sampling import ddm_sample
from utils.visualization import tensor_to_pil, create_image_grid

# Setup logger
logger = setup_logger("DDMCoordinator")

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
        self.device = torch.device(f"cuda:{rank}")
        
        # Initialize logger
        self.logger = setup_logger(name="DDMCoordinator", rank=rank)
        
        # Log starting information
        if is_main_process():
            log_training_start(self.logger, config)
        
        # Initialize WandB if specified
        if is_main_process() and hasattr(config, 'use_wandb') and config.use_wandb:
            self.run = init_wandb(config)
        else:
            self.run = None
        
        # Initialize cluster manager
        self.init_cluster_manager()
        
        # Initialize data loaders
        self.init_data_loaders()
        
        # Initialize models
        self.init_models()
        
        # Initialize optimizers
        self.init_optimizers()
        
        # Initialize diffusion components
        self.alphas, self.alpha_bar, self.betas = get_alphas_and_betas(
            num_timesteps=1000,
            schedule_type='cosine'
        )
        
        # Initialize flow matcher
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma,
            loss_type=config.loss_type
        )
        
        # Initialize VAE and CLIP
        self.vae = VAEWrapper(self.device, config)
        self.clip = CLIPTextEncoder(self.device, config)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricCalculator()
        
        # Initialize step counter
        self.current_step = 0
        
        self.logger.info("DDM Training Coordinator initialized successfully")
        
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
            
            self.logger.info("Cluster manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize cluster manager: {str(e)}")
            raise
            
    def init_data_loaders(self):
        """Initialize data loaders for training"""
        self.logger.info("Initializing data loaders")
        
        try:
            # Create feature dataset for clustering
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
            
            # Perform initial clustering
            self.cluster_manager.perform_clustering(feature_loader)
            
            # Get cluster assignments
            cluster_labels = self.cluster_manager.get_clusters()
            
            # Create main dataset
            self.dataset = DDMDataset(
                root_dir=self.config.dataset_path,
                cluster_labels=cluster_labels,
                include_metadata=True
            )
            
            # Create per-expert data loaders
            from data.loader import create_expert_bucket_loaders
            self.expert_loaders = create_expert_bucket_loaders(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Create router data loader
            self.router_loader = create_loader(
                dataset=self.dataset,
                config=self.config,
                is_train=True,
                distributed=(self.world_size > 1),
                rank=self.rank,
                world_size=self.world_size
            )
            
            self.logger.info(f"Data loaders initialized with {len(self.dataset)} images")
        except Exception as e:
            self.logger.error(f"Failed to initialize data loaders: {str(e)}")
            raise
            
    def perform_reclustering(self):
        """Perform reclustering of the dataset"""
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
            
            # Recreate per-expert data loaders
            from data.loader import create_expert_bucket_loaders
            self.expert_loaders = create_expert_bucket_loaders(
                dataset=self.dataset,
                config=self.config,
                world_size=self.world_size,
                rank=self.rank
            )
            
            # Handle expert parameter migration
            for old_idx, new_idx in cluster_mapping.items():
                if old_idx != new_idx:
                    self.migrate_expert_data(old_idx, new_idx)
            
            self.logger.info("Reclustering completed successfully")
            
            # Log clustering metrics if using wandb
            if self.run is not None and is_main_process():
                # Log cluster distribution
                cluster_counts = np.bincount(new_labels, minlength=self.config.num_experts)
                cluster_metrics = {f"cluster_{i}_count": count for i, count in enumerate(cluster_counts)}
                log_metrics(cluster_metrics, step=self.current_step, prefix="clustering")
            
        except Exception as e:
            self.logger.error(f"Failed to perform reclustering: {str(e)}")
            raise
            
    def init_models(self):
        """Initialize models for training"""
        self.logger.info("Initializing models")
        
        try:
            # Initialize experts
            self.experts = {}
            for expert_idx in range(self.config.num_experts):
                # Only initialize experts assigned to this process
                if expert_idx % self.world_size == self.rank:
                    self.experts[expert_idx] = ExpertTrainer(
                        expert_idx=expert_idx,
                        config=self.config,
                        device=self.device,
                        rank=self.rank,
                        world_size=self.world_size
                    )
            
            # Initialize router
            self.router_trainer = RouterTrainer(
                config=self.config,
                device=self.device,
                rank=self.rank,
                world_size=self.world_size
            )
            
            self.logger.info(f"Initialized {len(self.experts)} experts on this process")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise
            
    def init_optimizers(self):
        """Initialize optimizers for models"""
        # Optimizers are initialized in the respective trainers
        pass
            
    def train_experts(self, step):
        """Train expert models for one step"""
        self.current_step = step
        total_loss = 0.0
        
        # Train each expert assigned to this process
        for expert_idx, expert in self.experts.items():
            # Skip if no data loader for this expert
            if expert_idx not in self.expert_loaders:
                continue
                
            # Get batch
            try:
                batch = next(iter(self.expert_loaders[expert_idx]))
            except (StopIteration, IndexError):
                continue
                
            # Train expert
            loss = expert.train_step(batch)
            total_loss += loss
            
        # Average loss across experts
        avg_loss = total_loss / max(1, len(self.experts))
        
        return avg_loss
            
    def train_router(self, step):
        """Train router model for one step"""
        self.current_step = step
        
        # Skip router training if no data loader
        if not hasattr(self, 'router_loader') or not self.router_loader:
            return 0.0
            
        # Get batch
        try:
            batch = next(iter(self.router_loader))
        except (StopIteration, IndexError):
            return 0.0
            
        # Train router
        loss = self.router_trainer.train_epoch(self.router_loader)
        
        return loss
            
    def run_validation(self, step):
        """Run validation at current step"""
        self.current_step = step
        self.logger.info(f"Running validation at step {step}")
        
        # Skip if not main process
        if not is_main_process():
            return
            
        try:
            # Sample images
            with torch.no_grad():
                # Setup for sampling
                sample_shape = (
                    getattr(self.config, 'validation_samples', 4),
                    self.config.latent_channels,
                    self.config.image_size // 8,
                    self.config.image_size // 8
                )
                
                # Get text conditioning
                prompts = getattr(self.config, 'validation_prompts', ['a photo of a cat'])
                if isinstance(prompts, str):
                    prompts = [prompts]
                
                # Ensure we have enough prompts
                while len(prompts) < sample_shape[0]:
                    prompts.extend(prompts[:sample_shape[0] - len(prompts)])
                prompts = prompts[:sample_shape[0]]
                
                # Encode prompts
                text_embeddings, uncond_embeddings = self.clip.encode_with_uncond(prompts)
                
                # Sample from models
                latents = ddm_sample(
                    router=self.router_trainer.router,
                    experts=[expert.expert for expert in self.experts.values()],
                    shape=sample_shape,
                    steps=getattr(self.config, 'inference_steps', 50),
                    top_k=getattr(self.config, 'validation_topk', 1),
                    device=self.device,
                    cfg_scale=getattr(self.config, 'cfg_scale', 7.5),
                    text_embeddings=text_embeddings,
                    uncond_embeddings=uncond_embeddings
                )
                
                # Decode latents
                images = self.vae.decode(latents)
                
                # Log images
                if self.run is not None:
                    log_images(
                        images=images,
                        captions=prompts,
                        step=step,
                        prefix="validation"
                    )
                
                # Create grid for visualization
                grid = create_image_grid(images)
                
                # Save grid
                os.makedirs(self.config.sample_dir, exist_ok=True)
                grid.save(os.path.join(self.config.sample_dir, f"step_{step}.png"))
                
            self.logger.info(f"Validation completed, samples saved to {self.config.sample_dir}")
        except Exception as e:
            self.logger.error(f"Failed to run validation: {str(e)}")
            
    def save_sharded_checkpoints(self, step):
        """Save checkpoints for all models"""
        self.current_step = step
        self.logger.info(f"Saving checkpoints at step {step}")
        
        try:
            # Create checkpoint directory
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            
            # Save expert checkpoints
            for expert_idx, expert in self.experts.items():
                expert.save_checkpoint(
                    save_dir=self.config.checkpoint_dir,
                    step=step
                )
                
            # Save router checkpoint
            self.router_trainer.save_checkpoint(
                save_dir=self.config.checkpoint_dir,
                step=step
            )
            
            # Save coordinator state
            if is_main_process():
                coordinator_state = {
                    'step': step,
                    'config': vars(self.config)
                }
                save_sharded(
                    checkpoint=coordinator_state,
                    path=os.path.join(self.config.checkpoint_dir, f"coordinator_step{step}.pt")
                )
                
            self.logger.info(f"Checkpoints saved to {self.config.checkpoint_dir}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoints: {str(e)}")
            
    def log_sharded_metrics(self, step, expert_loss, router_loss):
        """Log metrics using centralized logging"""
        self.current_step = step
        
        # Skip if not using wandb
        if self.run is None:
            return
            
        # Collect metrics
        metrics = {
            'expert_loss': expert_loss,
            'router_loss': router_loss,
            'lr': self.get_current_lr()
        }
        
        # Log metrics
        log_metrics(metrics, step=step)
        
    def calculate_fid(self, generated_samples, real_dataset):
        """Calculate FID score between generated and real samples"""
        return self.metrics_calculator.fid(real_dataset, generated_samples)
        
    def get_current_lr(self):
        """Get current learning rate"""
        # Get learning rate from first expert
        if self.experts:
            expert = next(iter(self.experts.values()))
            return expert.get_learning_rate()
        return 0.0
        
    def log_to_wandb(self, step, fid, samples):
        """Log metrics and samples to W&B"""
        # Skip if not using wandb
        if self.run is None:
            return
            
        # Log metrics
        metrics = {
            'fid': fid,
            'step': step
        }
        log_metrics(metrics, step=step)
        
        # Log samples
        log_images(
            images=samples,
            step=step,
            prefix="samples"
        )
        
    def train_distilled_model(self):
        """Train distilled model from experts"""
        if not hasattr(self.config, 'do_distillation') or not self.config.do_distillation:
            self.logger.info("Skipping distillation as it's disabled in config")
            return
            
        self.logger.info("Training distilled model")
        
        try:
            # Initialize distilled model
            distilled_model = ExpertDiT(self.config).to(self.device)
            
            # Create distiller
            distiller = DiffusionDistiller(
                teacher={
                    'router': self.router_trainer.router,
                    'experts': [expert.expert for expert in self.experts.values()]
                },
                student=distilled_model,
                num_train_timesteps=1000,
                lr=getattr(self.config, 'distill_lr', 1e-5),
                warmup_ratio=0.05
            )
            
            # Create dataset for distillation
            from data.dataset import DDMDataset
            from data.loader import create_loader
            
            distill_dataset = DDMDataset(
                root_dir=self.config.dataset_path,
                include_metadata=False
            )
            
            distill_loader = create_loader(
                dataset=distill_dataset,
                config=self.config,
                is_train=True,
                distributed=(self.world_size > 1),
                rank=self.rank,
                world_size=self.world_size
            )
            
            # Train distilled model
            distiller.train(
                loader=distill_loader,
                epochs=getattr(self.config, 'distill_epochs', 10)
            )
            
            # Save distilled model
            if is_main_process():
                save_model_checkpoint(
                    model=distilled_model,
                    path=os.path.join(self.config.checkpoint_dir, "distilled_model.pt"),
                    metadata={'step': self.current_step}
                )
                
            self.logger.info("Distillation completed, model saved")
        except Exception as e:
            self.logger.error(f"Failed to train distilled model: {str(e)}")
            
    def needs_reclustering(self, step):
        """Check if reclustering is needed at current step"""
        if not hasattr(self.config, 'recluster_interval'):
            return False
            
        return step > 0 and step % self.config.recluster_interval == 0
        
    def get_expert(self, idx):
        """Get expert by index"""
        if idx in self.experts:
            return self.experts[idx]
            
        # Broadcast expert from the process that owns it
        owner_rank = idx % self.world_size
        if is_main_process():
            self.logger.info(f"Requesting expert {idx} from rank {owner_rank}")
            
        expert = None
        if self.rank == owner_rank:
            expert = self.experts[idx]
            
        return broadcast_object(expert, src=owner_rank)
        
    def migrate_expert_data(self, old_idx, new_idx):
        """Migrate expert data during reclustering"""
        self.logger.info(f"Migrating expert data from {old_idx} to {new_idx}")
        
        # Get source expert
        source_expert = self.get_expert(old_idx)
        
        # If we don't own the target expert, no need to do anything
        if new_idx % self.world_size != self.rank:
            return
            
        # If we don't have the target expert, create it
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
        target_expert.expert.load_state_dict(source_expert.expert.state_dict())
        
        self.logger.info(f"Expert data migrated from {old_idx} to {new_idx}")
