"""Distillation for Decentralized Diffusion Models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from models.dit import ExpertDiT
from utils.logging import log_metrics, log_images
from utils.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from utils.expert_cache import ExpertCacheManager

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
        
        # Create expert cache manager for memory efficiency
        self.expert_cache = ExpertCacheManager(config, device)
        
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
        
        # Setup AMP scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
        
        # Cache for expert outputs to avoid duplicate computations
        self.expert_output_cache = {}
        
        logger.info(f"Initialized DiffusionDistiller with {len(experts)} experts")
        
    def setup_data_loader(self):
        """Create data loader for distillation training with balanced cluster representation"""
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
        """
        Get balanced indices from dataset for distillation
        
        This ensures all clusters are equally represented in the distillation training
        """
        # Get all cluster assignments
        cluster_assignments = self.dataset.cluster_assignments
        
        # Count samples per cluster
        unique_clusters, cluster_counts = torch.unique(
            torch.tensor(cluster_assignments), 
            return_counts=True
        )
        
        # Calculate number of samples per cluster
        num_clusters = len(unique_clusters)
        samples_per_cluster = num_samples // num_clusters
        
        # Create balanced indices
        balanced_indices = []
        
        # Get indices for each cluster
        for cluster_idx in unique_clusters:
            # Get indices for this cluster
            cluster_indices = torch.where(torch.tensor(cluster_assignments) == cluster_idx)[0]
            
            # Randomly select samples from this cluster
            if len(cluster_indices) <= samples_per_cluster:
                # Use all samples if we have fewer than needed
                selected_indices = cluster_indices
            else:
                # Randomly select samples
                perm = torch.randperm(len(cluster_indices))
                selected_indices = cluster_indices[perm[:samples_per_cluster]]
                
            balanced_indices.extend(selected_indices.tolist())
            
        # Shuffle indices
        perm = torch.randperm(len(balanced_indices))
        balanced_indices = [balanced_indices[i] for i in perm]
        
        logger.info(f"Created balanced dataset with {len(balanced_indices)} samples across {num_clusters} clusters")
        
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
                
                # Train with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    # Forward through student (distilled) model
                    self.distilled_model.train()
                    student_pred = self.distilled_model(x_t, t_indices, text_embeds)
                    
                    # Forward through teacher (expert) models
                    # Create target tensor same shape as student prediction
                    teacher_pred = torch.zeros_like(student_pred)
                    
                    # Get unique cluster labels in batch for efficient expert loading
                    unique_clusters, cluster_counts = torch.unique(cluster_labels, return_counts=True)
                    
                    # Process each unique cluster with the corresponding expert
                    for cluster_idx in unique_clusters:
                        # Get expert model
                        try:
                            # Use expert cache manager to efficiently retrieve expert
                            expert_idx = cluster_idx.item()
                            expert = self._get_expert(expert_idx)
                            
                            if expert is None:
                                logger.warning(f"Expert {expert_idx} not found, skipping")
                                continue
                                
                            # Get mask for all samples from this cluster
                            cluster_mask = (cluster_labels == cluster_idx)
                            
                            # Skip if no samples from this cluster
                            if not cluster_mask.any():
                                continue
                                
                            # Get all samples from this cluster
                            cluster_x_t = x_t[cluster_mask]
                            cluster_t_indices = t_indices[cluster_mask]
                            cluster_text = text_embeds[cluster_mask] if text_embeds is not None else None
                            
                            # Forward through expert
                            with torch.no_grad():
                                expert_output = expert(cluster_x_t, cluster_t_indices, cluster_text)
                                
                            # Fill teacher predictions for this cluster
                            teacher_pred[cluster_mask] = expert_output
                            
                        except Exception as e:
                            logger.error(f"Error processing expert {cluster_idx.item()}: {str(e)}")
                            # Continue with next expert
                    
                    # Compute distillation loss
                    if self.config.distill_loss_type == "mse":
                        loss = F.mse_loss(student_pred, teacher_pred)
                    elif self.config.distill_loss_type == "huber":
                        loss = F.huber_loss(student_pred, teacher_pred)
                    elif self.config.distill_loss_type == "l1":
                        loss = F.l1_loss(student_pred, teacher_pred)
                    else:
                        loss = F.mse_loss(student_pred, teacher_pred)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.distilled_model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update EMA model
                self._update_ema_model()
                
                # Update loss stats
                epoch_loss += loss.item()
                steps += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "clusters": len(unique_clusters)
                })
                
                # Clear cache after each step to save memory
                self._clear_expert_cache()
            
            # Log epoch stats
            avg_epoch_loss = epoch_loss / max(1, steps)
            logger.info(f"Epoch {epoch+1}/{self.config.distill_epochs}, Loss: {avg_epoch_loss:.6f}")
            
            # Save checkpoint if this is the best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"Saved best model with loss {self.best_loss:.6f}")
                
        # Save final model
        self.save_checkpoint(self.config.distill_epochs - 1, is_final=True)
        logger.info("Distillation training completed")
        
        return avg_epoch_loss
    
    def _update_ema_model(self):
        """Update EMA model with current model weights"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.distilled_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _get_expert(self, expert_idx):
        """Get expert model using cache manager"""
        # Check if expert is available
        if expert_idx not in self.experts:
            return None
            
        # Define expert builder function for cache manager
        def expert_builder(idx):
            return self.experts[idx]
            
        # Use expert cache manager to retrieve expert
        try:
            return self.expert_cache.get_expert(expert_idx, expert_builder)
        except Exception as e:
            logger.error(f"Error getting expert {expert_idx}: {str(e)}")
            return None
    
    def _clear_expert_cache(self):
        """Clear expert cache to free memory"""
        self.expert_output_cache.clear()
        self.expert_cache.clear_cache()
        torch.cuda.empty_cache()
    
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
