import torch
import torch.nn as nn

class BaseTrainer(nn.Module):
    """Shared training logic from paper implementations"""
    
    def __init__(self, config, device, rank):
        super().__init__()
        self.config = config
        self.device = device
        self.rank = rank
        
    def train_step(self, batch):
        """Paper-standard training step with gradient clipping"""
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        self.optimizer.step()
        return loss.item()
    
    def compute_loss(self, batch):
        """To be implemented by subclasses"""
        raise NotImplementedError