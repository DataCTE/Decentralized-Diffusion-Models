"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, CPUOffload
from torch.nn import functional as F

from models.router import RouterModel

class RouterTrainer:
    """Trainer for the router model in DDM"""
    def __init__(self, config, device, rank, world_size=None):
        # Initialize parameters
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size or 1
        
        # Create base router model
        base_router = RouterModel(config).to(device)
        
        # Apply FSDP if world_size > 1
        if self.world_size > 1:
            # Configure FSDP settings based on config
            # Sharding strategy
            if config.fsdp_sharding_strategy == "FULL_SHARD":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            elif config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            else:
                sharding_strategy = ShardingStrategy.FULL_SHARD
                
            # CPU offload
            cpu_offload = CPUOffload(offload_params=config.fsdp_cpu_offload)
            
            # Backward prefetch
            if config.fsdp_backward_prefetch == "BACKWARD_PRE":
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            elif config.fsdp_backward_prefetch == "BACKWARD_POST":
                backward_prefetch = BackwardPrefetch.BACKWARD_POST
            else:
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
                
            # Auto wrap policy
            if config.fsdp_auto_wrap_policy == "DEFAULT":
                auto_wrap_policy = default_auto_wrap_policy
            elif config.fsdp_auto_wrap_policy == "SIZE_BASED":
                auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=config.fsdp_min_num_params)
            else:
                auto_wrap_policy = default_auto_wrap_policy
            
            # Apply FSDP to shard model across all GPUs
            self.router = FSDP(
                base_router,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=True  # Allow easier parameter access
            )
            
            if rank == 0:
                print(f"Initialized SHARDED Router across {self.world_size} GPUs")
        else:
            # Just use the base model without FSDP
            self.router = base_router
            
        # Paper-recommended optimizer settings
        self.optimizer = AdamW8bit(
            self.router.parameters(),
            lr=config.router_learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Paper-recommended learning schedule
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/self.config.warmup_steps, 1.0) 
            if step < self.config.warmup_steps 
            else 0.5*(1 + math.cos(math.pi*(step - self.config.warmup_steps)/self.config.total_steps))
        )

    def train_epoch(self, loader):
        """
        Implements Algorithm 1 from paper (router training)
        
        This trains the router model to predict which expert should handle
        each sample, as described in Section 3.3 of the paper.
        """
        # Add gradient isolation
        for param in self.router.parameters():
            param.requires_grad_(False)  # Freeze first
        
        # Only unfreeze router-specific params
        for name, param in self.router.named_parameters():
            if "classifier" in name or "cls_token" in name:
                param.requires_grad_(True)
        
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
            
            # Add confidence thresholding
            if self.config.router_confidence_threshold > 0:
                probs = torch.softmax(logits, dim=1)
                max_prob = probs.max(dim=1)[0]
                mask = (max_prob > self.config.router_confidence_threshold).float()
                loss = (loss * mask).mean()
            
            # Optimization (Section 4.1)
            # The paper uses AdamW with weight decay
            self.optimizer.zero_grad()
            loss.backward()
            # Paper-recommended gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.router.parameters(), 
                max_norm=self.config.max_grad_norm,  # Should be 1.0 in config
                norm_type=2.0
            )
            self.optimizer.step()
            self.lr_scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            num_batches += 1
        
        # Return average loss over the epoch
        return total_loss / num_batches

    def calibrate_confidence(self, val_loader):
        """Paper-recommended temperature scaling (Section 3.3)"""
        # Freeze all parameters except temperature
        for param in self.router.parameters():
            param.requires_grad = False
        self.router.temperature.requires_grad = True
        
        optimizer = torch.optim.LBFGS([self.router.temperature], lr=0.01)
        
        def eval_fn():
            optimizer.zero_grad()
            loss = 0
            for batch in val_loader:
                images = batch["image"].to(self.device)
                clusters = batch["cluster"].to(self.device)
                t = torch.rand(images.size(0), device=self.device)
                noise = torch.randn_like(images)
                xt = torch.cos(t * math.pi/2)[:,None,None,None] * images + \
                     torch.sin(t * math.pi/2)[:,None,None,None] * noise
                    
                logits = self.router(xt, t)
                loss += F.cross_entropy(logits, clusters)
            loss.backward()
            return loss
        
        # Run L-BFGS optimization
        for _ in range(100):
            optimizer.step(eval_fn)
