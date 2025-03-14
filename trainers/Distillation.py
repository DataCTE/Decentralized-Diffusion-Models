"""Distillation for Decentralized Diffusion Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset

from models.dit import ExpertDiT
from utils.logging import log_metrics, log_images
from utils.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas

logger = logging.getLogger(__name__)

class DiffusionDistiller:
    """
    Knowledge distillation from expert ensemble to single model
    
    Paper Section 3.6: The sparse DDM model can be distilled into a dense model.
    """
    def __init__(self, config, experts, router, dataset, device, rank=0):
        """
        Initialize distiller
        
        Args:
            config: Training configuration
            experts: List of expert models
            router: Router model
            dataset: Full dataset with cluster labels
            device: Device for training
            rank: Process rank
        """
        self.config = config
        self.device = device
        self.rank = rank
        self.experts = experts
        self.router = router
        self.dataset = dataset
        
        # Initialize distilled model
        self.distilled_model = ExpertDiT(config).to(device)
        
        # Paper-recommended optimizer settings for distillation
        self.optimizer = torch.optim.AdamW(
            self.distilled_model.parameters(),
            lr=config.distill_lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # Cosine learning rate scheduler
        self.total_steps = config.distill_epochs * (config.distill_samples // config.distill_batch_size)
        self.warmup_steps = int(self.total_steps * 0.1)  # 10% warmup
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/self.warmup_steps, 1.0) 
            if step < self.warmup_steps 
            else 0.5*(1 + math.cos(math.pi*(step - self.warmup_steps)/(self.total_steps - self.warmup_steps)))
        )
        
        # Initialize flow matcher for loss computation
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma,
            loss_type=config.loss_type
        )
        
        # Get diffusion parameters
        self.alphas, self.alpha_bar, _ = get_alphas_and_betas()
        
        # Initialize EMA model for stable results
        self.ema_model = ExpertDiT(config).to(device)
        self.ema_model.load_state_dict(self.distilled_model.state_dict())
        self.ema_decay = config.ema_decay
        
        # Track metrics
        self.best_loss = float('inf')
        
    def setup_data_loader(self):
        """Create data loader for distillation training"""
        # For distillation, we can use a balanced subset of the data
        balanced_indices = self._get_balanced_indices(self.config.distill_samples)
        distill_subset = Subset(self.dataset, balanced_indices)
        
        # Create data loader for distillation
        distill_loader = DataLoader(
            distill_subset,
            batch_size=self.config.distill_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        return distill_loader
        
    def _get_balanced_indices(self, num_samples):
        """Get balanced indices from the dataset based on cluster assignments"""
        cluster_labels = self.dataset.cluster_assignments
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)
        
        # Samples per cluster (ensure balanced representation)
        samples_per_cluster = num_samples // num_clusters
        
        balanced_indices = []
        for cluster in unique_clusters:
            # Get indices for this cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            # If not enough samples in this cluster, sample with replacement
            if len(cluster_indices) < samples_per_cluster:
                sampled_indices = np.random.choice(
                    cluster_indices, 
                    size=samples_per_cluster, 
                    replace=True
                )
            else:
                # Otherwise sample without replacement
                sampled_indices = np.random.choice(
                    cluster_indices, 
                    size=samples_per_cluster, 
                    replace=False
                )
                
            balanced_indices.extend(sampled_indices)
            
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        
        return balanced_indices
    
    def update_ema(self):
        """Update EMA model parameters"""
        with torch.no_grad():
            for param, ema_param in zip(
                self.distilled_model.parameters(), 
                self.ema_model.parameters()
            ):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data
    
    def train(self):
        """Train distilled model from expert ensemble"""
        logger.info("Starting distillation training")
        
        # Setup data loader
        train_loader = self.setup_data_loader()
        
        # Track losses
        total_loss = 0.0
        count = 0
        
        # Training loop
        for epoch in range(self.config.distill_epochs):
            epoch_loss = 0.0
            steps = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{self.config.distill_epochs}")
            
            for batch in pbar:
                # Get data
                images = batch["image"].to(self.device)
                cluster_labels = batch["cluster"].to(self.device)
                
                # Get batch size
                batch_size = images.shape[0]
                
                # Sample random timesteps t ∈ [0, 1]
                t_indices = torch.randint(0, 1000, (batch_size,), device=self.device)
                t = t_indices.float() / 1000.0  # Normalize to [0, 1]
                
                # Sample random noise
                noise = torch.randn_like(images)
                
                # Forward process to get x_t
                alpha_t = torch.sqrt(self.alpha_bar[t_indices])
                sigma_t = torch.sqrt(1 - self.alpha_bar[t_indices])
                
                x_t = alpha_t.view(-1, 1, 1, 1) * images + sigma_t.view(-1, 1, 1, 1) * noise
                
                # Get text conditions if available
                text_embeds = None
                if "caption_embedding" in batch:
                    text_embeds = batch["caption_embedding"].to(self.device)
                
                # Forward through student (distilled) model
                self.distilled_model.train()
                student_pred = self.distilled_model(x_t, t_indices, text_embeds)
                
                # Forward through teacher (expert) models
                # Each batch item uses its corresponding expert as teacher
                with torch.no_grad():
                    # Create target tensor same shape as student prediction
                    teacher_pred = torch.zeros_like(student_pred)
                    
                    # For each item in batch, use the expert assigned to its cluster
                    for i in range(batch_size):
                        # Get expert index
                        expert_idx = cluster_labels[i].item()
                        
                        # Get the expert
                        expert = self.experts[expert_idx]
                        
                        # Forward through expert
                        if text_embeds is not None:
                            # Include text conditioning
                            expert_output = expert(
                                x_t[i:i+1], 
                                t_indices[i:i+1], 
                                text_embeds[i:i+1]
                            )
                        else:
                            # No text conditioning
                            expert_output = expert(
                                x_t[i:i+1], 
                                t_indices[i:i+1]
                            )
                        
                        # Store expert output
                        teacher_pred[i:i+1] = expert_output
                
                # Compute distillation loss
                loss = F.mse_loss(student_pred, teacher_pred)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.distilled_model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                
                # Update EMA model
                self.update_ema()
                
                # Track loss
                epoch_loss += loss.item()
                steps += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
            
            # Calculate average epoch loss
            epoch_loss /= steps
            total_loss += epoch_loss
            count += 1
            
            # Log epoch metrics
            if self.rank == 0:
                log_metrics({
                    "distill/epoch": epoch,
                    "distill/loss": epoch_loss,
                    "distill/lr": self.scheduler.get_last_lr()[0]
                }, step=epoch)
                
                # Save best model
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.save_checkpoint(epoch, is_best=True)
                
                # Regular checkpoint
                if (epoch + 1) % 5 == 0 or epoch == self.config.distill_epochs - 1:
                    self.save_checkpoint(epoch)
        
        # Calculate final average loss
        avg_loss = total_loss / count
        
        logger.info(f"Distillation complete, average loss: {avg_loss:.6f}")
        
        # Save final model
        self.save_checkpoint(self.config.distill_epochs - 1, is_final=True)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False, is_final=False):
        """Save distilled model checkpoint"""
        # Create output directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Base checkpoint path
        if is_final:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "distilled_model_final.pt")
        elif is_best:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "distilled_model_best.pt")
        else:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"distilled_model_epoch{epoch}.pt")
        
        # Save model
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.distilled_model.state_dict(),
            "ema_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")},
            "loss": self.best_loss
        }, checkpoint_path)
        
        logger.info(f"Saved distilled model checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load distilled model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.distilled_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load EMA model state if available
        if "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            # Fall back to regular model state
            self.ema_model.load_state_dict(checkpoint["model_state_dict"])
            
        # Load optimizer and scheduler if training
        if hasattr(self, "optimizer") and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load loss
        if "loss" in checkpoint:
            self.best_loss = checkpoint["loss"]
            
        logger.info(f"Loaded distilled model checkpoint from {checkpoint_path}")
        
        # Return epoch for resuming
        return checkpoint.get("epoch", 0)
