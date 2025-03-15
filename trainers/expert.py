"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, CPUOffload

from models.dit import ExpertDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.logging import logger



class ExpertTrainer(BaseTrainer):
    """Trainer for expert DiT models in DDM"""
    def __init__(self, expert_idx, config, device, rank, world_size):
        # Paper-recommended initialization (section 4.1)
        super().__init__(config, device, rank)
        self.expert_idx = expert_idx  # Store the expert index for identification
        self.world_size = world_size
        
        # Create base model
        base_expert = ExpertDiT(config).to(device)
        
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
            auto_wrap_policy = lambda_auto_wrap_policy
        elif config.fsdp_auto_wrap_policy == "SIZE_BASED":
            auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=config.fsdp_min_num_params)
        else:
            auto_wrap_policy = lambda_auto_wrap_policy
            
        # Apply FSDP with proper isolation
        self.expert = FSDP(
            base_expert,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            # Properly isolate parameters - don't use ignored_parameters
            param_init_fn=lambda module: module.to_empty(device=torch.cuda.current_device(), recurse=False)
        )
        
        # Paper-specified optimizer settings - use Adam with FSDP
        self.optimizer = AdamW8bit(
            self.expert.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        # Paper-defined components
        self.flow_matcher = DecentralizedFlowMatcher(
            sigma=config.sigma, 
            loss_type=config.loss_type
        )
        self.vae = VAEWrapper(device, config)
        self.clip = CLIPTextEncoder(device, config)
        
        # Precompute diffusion schedule as in paper appendix
        self.alphas, self.alpha_bar, _ = get_alphas_and_betas()
        
        # Learning rate scheduler
        warmup_steps = int(0.05 * config.num_steps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/warmup_steps, 1.0) 
            if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (config.num_steps - warmup_steps)))
        )
        
        # Log initialization of this expert
        if rank == 0:
            print(f"Initialized SHARDED Expert {expert_idx} across {world_size} GPUs")

    def compute_loss(self, batch):
        return self.flow_matcher.compute_loss(
            self.expert(batch['x_t'], batch['t']),
            batch['x0'],
            batch['t']
        )

    def train_step(self, batch):
        """
        Implements Algorithm 1 from paper with proper gradient isolation
        
        This trains an expert model using the flow matching objective
        as described in Section 3.2 of the paper.
        """
        images = batch["image"].to(self.device)
        
        # Use mixed precision training if configured
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # VAE encoding (paper section 4.1)
            latents = self.vae.encode(images)
            
            # Sample random timesteps t ∈ [0, 1] (Section 3.2)
            t_indices = torch.randint(0, 1000, (latents.size(0),), device=self.device)
            t = t_indices.float() / 1000.0  # Normalize to [0, 1]
            
            # Sample random noise (Section 3.2)
            noise = torch.randn_like(latents)
            
            # Forward process using cosine schedule (Section 3.2)
            alpha_t = torch.cos((t + 0.008)/1.008 * math.pi/2).pow(2)
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            latent_t = alpha_t * latents + sigma_t * noise
            
            # Text conditioning (paper section 4.1)
            text_embeds = self.clip.encode(batch["caption"])
            
            # Expert prediction of flow field u_t(x_t) (Equation 6)
            pred_flow = self.expert(latent_t, t_indices, text_embeds)
            
            # The target flow field v_t(x_t) (Equation 4)
            target_flow = self.flow_matcher.compute_flow_matching_target(
                latents, latent_t, t
            )
            
            # Flow matching loss (Equation 7)
            loss = self.flow_matcher.compute_flow_matching_loss(
                pred_flow, target_flow
            )
        
        # Optimize with gradient isolation
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        
        # Use isolated gradient norm calculation for proper clipping
        with torch.no_grad():
            # Get this expert's parameters only
            expert_params = list(self.expert.parameters())
            
            # Calculate norm for only this expert's gradients
            if expert_params and expert_params[0].grad is not None:
                grad_norm = torch.norm(
                    torch.stack([
                        torch.norm(p.grad.detach(), 2, dim=-1) 
                        for p in expert_params if p.grad is not None
                    ]), 
                    2
                )
                
                # Apply clipping if norm exceeds threshold
                clip_coef = self.config.max_grad_norm / (grad_norm + 1e-6)
                if clip_coef < 1:
                    for p in expert_params:
                        if p.grad is not None:
                            p.grad.detach().mul_(clip_coef)
                
                if self.rank == 0 and torch.rand(1).item() < 0.01:  # Log occasionally
                    logger.debug(f"Expert {self.expert_idx} grad norm: {grad_norm:.4f}, clip: {clip_coef < 1}")
        
        # Finish optimization
        scaler.step(self.optimizer)
        scaler.update()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        return loss.item()
    
    def save_checkpoint(self, save_dir, step):
        """Save a checkpoint for the expert model using consolidated checkpoint utility"""
        # Create checkpoint path
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = f"{save_dir}/expert_{self.expert_idx}_step{step}.pt"
        
        # Create metadata
        metadata = {
            'expert_idx': self.expert_idx,
            'step': step,
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Save using the centralized utility
        return save_model_checkpoint(
            model=self.expert,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            metadata=metadata,
            is_fsdp=True
        )
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint for the expert model using consolidated checkpoint utility"""
        # Load using the centralized utility
        metadata = load_model_checkpoint(
            model=self.expert,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=True
        )
        
        return metadata
    
    def forward_diffuse(self, x0, t, noise):
        """Forward diffusion process with precomputed alpha_bar"""
        alpha_bar = self.alpha_bar.to(device=self.device, dtype=x0.dtype)
        
        # Extract alpha_bar for the specific timesteps
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
        
        # Apply forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def reset_parameters(self):
        """Reset expert parameters when assigned to a new cluster"""
        if self.rank == 0:
            print(f"Resetting parameters for expert {self.expert_idx}")
            
        # Reinitialize the model
        for module in self.expert.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                
        # Reset optimizer state
        self.optimizer = AdamW8bit(
            self.expert.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.adam_betas,
            weight_decay=self.config.weight_decay
        )
        
        # Reset scheduler
        warmup_steps = int(0.05 * self.config.num_steps)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/warmup_steps, 1.0) 
            if step < warmup_steps
            else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (self.config.num_steps - warmup_steps)))
        ) 

    def isolate_optimizer_state(self):
        """Prevent contamination of optimizer state across experts"""
        # Get current optimizer state
        state = self.optimizer.state_dict()
        
        # Identify parameters specific to this expert
        expert_params = {id(p): p for p in self.expert.parameters()}
        
        # Filter optimizer state to only include this expert's parameters
        if 'state' in state and state['state']:
            filtered_state = {}
            for param_id, param_state in state['state'].items():
                if param_id in expert_params:
                    filtered_state[param_id] = param_state
            
            # Replace with filtered state
            state['state'] = filtered_state
            
            # Load filtered state back
            self.optimizer.load_state_dict(state)
            
            if self.rank == 0:
                logger.info(f"Isolated optimizer state for expert {self.expert_idx}") 