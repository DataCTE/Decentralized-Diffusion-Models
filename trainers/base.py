import torch
import torch.nn as nn
import logging
import os
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
import math

logger = logging.getLogger(__name__)

class BaseTrainer(nn.Module):
    """Shared training logic from paper implementations"""
    
    def __init__(self, config, device, rank):
        super().__init__()
        self.config = config
        self.device = device
        self.rank = rank
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.current_step = 0
        
    def train_step(self, batch):
        """Paper-standard training step with gradient clipping"""
        if self.optimizer is None or self.model is None:
            raise ValueError("Trainer not fully initialized: optimizer or model is None")
            
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Update learning rate scheduler if available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        self.current_step += 1
        return loss.item()
    
    def compute_loss(self, batch):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement compute_loss")
        
    def save_checkpoint(self, save_dir, step=None):
        """Save trainer checkpoint"""
        if step is None:
            step = self.current_step
            
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = f"{save_dir}/{self.__class__.__name__}_step{step}.pt"
        
        # Create metadata
        metadata = {
            'step': step,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Use centralized checkpoint saving
        return save_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            metadata=metadata,
            is_fsdp=isinstance(self.model, torch.distributed.fsdp.FullyShardedDataParallel)
        )
        
    def load_checkpoint(self, checkpoint_path):
        """Load trainer checkpoint"""
        # Use centralized checkpoint loading
        metadata = load_model_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=isinstance(self.model, torch.distributed.fsdp.FullyShardedDataParallel),
            device=self.device
        )
        
        if metadata and 'step' in metadata:
            self.current_step = metadata['step']
            
        return metadata
        
    def get_learning_rate(self):
        """Get current learning rate"""
        if self.optimizer is None:
            return 0.0
            
        return self.optimizer.param_groups[0]['lr']
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
        return total_loss / max(1, num_batches)
        
    def evaluate(self, dataloader):
        """Evaluate model on validation data"""
        total_loss = 0.0
        num_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
        self.model.train()
        return total_loss / max(1, num_batches)
        
    def create_lr_scheduler(self, warmup_ratio=0.05, total_steps=None):
        """Create learning rate scheduler with warmup and cosine decay"""
        if self.optimizer is None:
            raise ValueError("Optimizer must be initialized before creating scheduler")
            
        if total_steps is None:
            total_steps = self.config.num_steps
            
        warmup_steps = int(warmup_ratio * total_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                return 0.5 * (1 + torch.cos(
                    torch.tensor(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))
                ).item())
                
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        
        return self.lr_scheduler