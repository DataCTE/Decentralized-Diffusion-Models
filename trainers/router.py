"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit

from models.router import RouterModel

class RouterTrainer:
    """Trainer for the router model in DDM"""
    def __init__(self, config, device, rank):
        # Paper-specified router architecture (section 3.3)
        self.router = RouterModel(config).to(device)
        self.config = config
        self.device = device
        
        # Paper-recommended training setup
        self.optimizer = AdamW8bit(
            self.router.parameters(),
            lr=config.router_learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        """
        Implements Algorithm 1 from paper (router training)
        
        This trains the router model to predict which expert should handle
        each sample, as described in Section 3.3 of the paper.
        """
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            # Get images and cluster assignments (Section 3.3)
            # The cluster assignment k* is the ground truth for router training
            images = batch["image"].to(self.device)
            clusters = batch["cluster"].to(self.device)  # k* in Algorithm 1
            
            # Sample random timesteps t ∈ [0, 1] (Section 3.3)
            # The paper uses uniform sampling of t in [0, 1]
            t = torch.rand(images.size(0), device=self.device)
            
            # Sample random noise (Section 3.3)
            # ε ~ N(0, I) as in Algorithm 1
            noise = torch.randn_like(images)
            
            # Forward process using cosine schedule (Section 3.3)
            # x_t = alpha_t * x_0 + sigma_t * noise
            # This follows the cosine schedule in the paper
            alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            xt = alpha_t * images + sigma_t * noise
            
            # Router prediction (Equation 5 in the paper)
            # The router predicts which expert should handle this sample
            # z = rθ(xt, t) ∈ R^K where K is the number of experts
            logits = self.router(xt, t)
            
            # Cross-entropy loss for router (Section 3.3)
            # L_router = E_{x_0,t}[-log p_k*(x_t, t)]
            # where k* is the cluster assignment for x_0
            # This is implemented as cross-entropy between logits and cluster labels
            loss = self.criterion(logits, clusters)
            
            # Optimization (Section 4.1)
            # The paper uses AdamW with weight decay
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Return average loss over the epoch
        return total_loss / num_batches 