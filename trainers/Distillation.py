"""Distillation for Decentralized Diffusion Models"""

import torch
import torch.nn.functional as F
import math
import os
import logging
from tqdm import tqdm

from models.dit import ExpertDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
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
            dataset: Full dataset with expert assignments
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

    def train_distilled_model(self, train_loader, val_loader=None, save_dir=None):
        """
        Train distilled model with supervision from expert ensemble
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            save_dir: Directory to save checkpoints
            
        Returns:
            Trained distilled model
        """
        # Setup logging
        self.logger.info("Starting distillation training (Paper Section 3.6)")
        
        # Setup checkpoint directory
        if save_dir is None:
            save_dir = os.path.join(self.config.checkpoint_dir, "distilled")
        os.makedirs(save_dir, exist_ok=True)
        
        # Track best model
        best_model_path = os.path.join(save_dir, "distilled_model_best.pt")
        
        # Main training loop
        global_step = 0
        for epoch in range(self.config.distill_epochs):
            # Track metrics
            epoch_loss = 0.0
            num_batches = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{self.config.distill_epochs}")
            
            # Training loop
            self.distilled_model.train()
            for batch in pbar:
                # Paper Section 3.6: "We select the appropriate expert for each training example based on its cluster label."
                images = batch["image"].to(self.device)
                text_embeds = batch.get("text_embedding")
                if text_embeds is not None:
                    text_embeds = text_embeds.to(self.device)
                    
                # Get timesteps (random for each batch item)
                batch_size = images.shape[0]
                t = torch.rand(batch_size, device=self.device)
                timesteps = (t * 1000).long()
                
                # Forward pass with gradient computation
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    # Apply forward diffusion
                    # Use cosine schedule as in the paper
                    alpha_t = torch.cos(t.view(-1, 1, 1, 1) * math.pi/2)
                    sigma_t = torch.sin(t.view(-1, 1, 1, 1) * math.pi/2)
                    
                    # Sample noise
                    noise = torch.randn_like(images)
                    
                    # Create noisy images
                    noisy_images = alpha_t * images + sigma_t * noise
                    
                    # Get student predictions
                    student_pred = self.distilled_model(noisy_images, timesteps, text_embeds)
                    
                    # Get router predictions for this batch
                    with torch.no_grad():
                        router_logits = self.router(
                            noisy_images, 
                            timesteps,
                            text_embeds
                        )
                        expert_weights = F.softmax(router_logits, dim=-1)
                    
                    # Select top-k experts per example
                    topk_weights, topk_indices = torch.topk(
                        expert_weights, 
                        self.config.top_k,
                        dim=-1
                    )
                    
                    # Get predictions from top-k experts
                    teacher_pred = torch.zeros_like(student_pred)
                    for k in range(self.config.top_k):
                        # Get expert indices for this top-k position
                        expert_indices = topk_indices[:, k]
                        
                        # Gather predictions from relevant experts
                        expert_preds = []
                        for idx in range(batch_size):
                            expert = self.expert_cache.get_expert(
                                expert_indices[idx].item(),
                                lambda idx: self.experts[idx]
                            )
                            with torch.no_grad():
                                pred = expert(
                                    noisy_images[idx:idx+1],
                                    timesteps[idx:idx+1],
                                    text_embeds[idx:idx+1] if text_embeds else None
                                )
                            expert_preds.append(pred)
                        
                        # Combine predictions using router weights
                        expert_preds = torch.cat(expert_preds)
                        teacher_pred += topk_weights[:, k].view(-1,1,1,1) * expert_preds
                    
                    # Compute loss
                    loss = F.mse_loss(student_pred, teacher_pred)
                
                # Optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Apply gradient clipping
                if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.distilled_model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                # Update model
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update EMA model
                with torch.no_grad():
                    ema_decay = self.ema_decay
                    for param, ema_param in zip(
                        self.distilled_model.parameters(),
                        self.ema_model.parameters()
                    ):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": loss.item(),
                    "avg_loss": epoch_loss / num_batches,
                    "lr": self.scheduler.get_last_lr()[0]
                })
                
                # Validate periodically
                if val_loader is not None and global_step % self.config.distill_val_interval == 0:
                    val_loss = self.validate(val_loader)
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_model(best_model_path)
                        self.logger.info(f"New best model: val_loss={val_loss:.6f}")
                
                # Save checkpoint periodically
                if global_step % self.config.distill_save_interval == 0:
                    checkpoint_path = os.path.join(save_dir, f"distilled_model_step_{global_step}.pt")
                    self.save_model(checkpoint_path)
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            self.logger.info(f"Epoch {epoch+1}/{self.config.distill_epochs} - Loss: {avg_epoch_loss:.6f}")
            
            # Save epoch checkpoint
            epoch_path = os.path.join(save_dir, f"distilled_model_epoch_{epoch+1}.pt")
            self.save_model(epoch_path)
            
        # Final validation
        if val_loader is not None:
            final_val_loss = self.validate(val_loader)
            self.logger.info(f"Final validation loss: {final_val_loss:.6f}")
            
            # Save best model if the final model is the best
            if final_val_loss < self.best_loss:
                self.best_loss = final_val_loss
                self.save_model(best_model_path)
                
        # Save final model
        final_path = os.path.join(save_dir, "distilled_model_final.pt")
        self.save_model(final_path)
        
        self.logger.info("Distillation training complete")
        return self.distilled_model
        
    def validate(self, val_loader):
        """
        Validate distilled model against expert ensemble
        
        Args:
            val_loader: DataLoader for validation
            
        Returns:
            Validation loss
        """
        self.distilled_model.eval()
        self.ema_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Get data
                images = batch["image"].to(self.device)
                clusters = batch["cluster"].to(self.device)
                text_embeds = batch.get("text_embedding")
                if text_embeds is not None:
                    text_embeds = text_embeds.to(self.device)
                
                # Get timesteps
                batch_size = images.shape[0]
                t = torch.rand(batch_size, device=self.device)
                timesteps = (t * 1000).long()
                
                # Apply forward diffusion
                alpha_t = torch.cos(t.view(-1, 1, 1, 1) * math.pi/2)
                sigma_t = torch.sin(t.view(-1, 1, 1, 1) * math.pi/2)
                noise = torch.randn_like(images)
                noisy_images = alpha_t * images + sigma_t * noise
                
                # Get student predictions from EMA model (more stable)
                student_pred = self.ema_model(noisy_images, timesteps, text_embeds)
                
                # Get router predictions
                router_logits = self.router(
                    noisy_images,
                    timesteps,
                    text_embeds
                )
                expert_weights = F.softmax(router_logits, dim=-1)
                
                # Select top expert for validation efficiency
                _, expert_indices = torch.max(expert_weights, dim=-1)
                
                # Get teacher predictions
                teacher_pred = torch.zeros_like(student_pred)
                for idx in range(batch_size):
                    expert_idx = expert_indices[idx].item()
                    expert = self.expert_cache.get_expert(
                        expert_idx,
                        lambda idx: self.experts[idx]
                    )
                    
                    x_idx = noisy_images[idx:idx+1]
                    t_idx = timesteps[idx:idx+1]
                    text_idx = text_embeds[idx:idx+1] if text_embeds else None
                    
                    expert_pred = expert(x_idx, t_idx, text_idx)
                    teacher_pred[idx:idx+1] = expert_pred
                
                # Compute loss
                loss = F.mse_loss(student_pred, teacher_pred)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.distilled_model.train()  # Set back to training mode
        
        return avg_loss
    
    def save_model(self, path):
        """Save distilled model with EMA weights"""
        try:
            torch.save({
                "model": self.distilled_model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_loss": self.best_loss,
                "config": self.config,
            }, path)
            self.logger.info(f"Saved model to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
