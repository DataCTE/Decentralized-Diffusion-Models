"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, CPUOffload

from models.dit import ExpertDiT
from utils.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder
from trainers.base import BaseTrainer

class ExpertTrainer(BaseTrainer):
    """Trainer for expert DiT models in DDM"""
    def __init__(self, expert_idx, config, device, rank, world_size):
        # Paper-recommended initialization (section 4.1)
        self.expert_idx = expert_idx  # Store the expert index for identification
        self.config = config
        self.device = device
        self.rank = rank
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
            auto_wrap_policy = default_auto_wrap_policy
        elif config.fsdp_auto_wrap_policy == "SIZE_BASED":
            auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=config.fsdp_min_num_params)
        else:
            auto_wrap_policy = default_auto_wrap_policy
            
        # Apply FSDP with explicit isolation
        self.expert = FSDP(
            base_expert,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            # Paper-mandated isolation parameters
            ignored_parameters=[],  # Remove incorrect base_router reference
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
        Implements Algorithm 1 from paper (expert training)
        
        This trains an expert model using the flow matching objective
        as described in Section 3.2 of the paper.
        """
        images = batch["image"].to(self.device)
        
        # Use mixed precision training if configured
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)
        
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # VAE encoding (paper section 4.1)
            # The paper uses a VAE to encode images into latent space
            latents = self.vae.encode(images)
            
            # Sample random timesteps t ∈ [0, 1] (Section 3.2)
            # Note: We sample from [0, 1000) and normalize to [0, 1]
            # The paper uses uniform sampling of t in [0, 1]
            t_indices = torch.randint(0, 1000, (latents.size(0),), device=self.device)
            t = t_indices.float() / 1000.0  # Normalize to [0, 1]
            
            # Sample random noise (Section 3.2)
            # ε ~ N(0, I) as in Algorithm 1
            noise = torch.randn_like(latents)
            
            # Forward process using cosine schedule (Section 3.2)
            # x_t = alpha_t * x_0 + sigma_t * noise
            # This follows the cosine schedule in the paper:
            # alpha_t = cos(t * pi/2), sigma_t = sin(t * pi/2)
            alpha_t = torch.cos((t + 0.008)/1.008 * math.pi/2).pow(2)
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            latent_t = alpha_t * latents + sigma_t * noise
            
            # Text conditioning (paper section 4.1)
            # The paper uses CLIP text embeddings for conditioning
            text_embeds = self.clip.encode(batch["caption"])
            
            # Expert prediction of flow field u_t(x_t) (Equation 6)
            # The expert predicts the flow field at the current timestep
            pred_flow = self.expert(latent_t, t_indices, text_embeds)
            
            # The target flow field v_t(x_t) (Equation 4)
            # This is the ground truth for flow matching objective
            target_flow = self.flow_matcher.compute_flow_matching_target(
                latents, latent_t, t
            )
            
            # Flow matching loss (Equation 7)
            # L_flow = E_{x_0,t}[|| u_t(x_t) - v_t(x_t) ||^2]
            loss = self.flow_matcher.compute_flow_matching_loss(
                pred_flow, target_flow
            )
        
        # Optimize using mixed precision
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        # Paper-recommended per-expert gradient clipping
        with self.expert.summon_full_params():
            torch.nn.utils.clip_grad_norm_(
                self.expert.parameters(), 
                max_norm=self.config.max_grad_norm,
                norm_type=2.0,
                # Prevent cross-expert norm calculation
                foreach=False  # Important for isolation
            )
        scaler.step(self.optimizer)
        scaler.update()
        
        if self.config.use_affinity_mask:
            mask = (batch['cluster'] == self.expert_idx).float()
            loss = mask * loss + (1-mask) * loss.detach()
        
        return loss.item()
    
    def save_checkpoint(self, save_dir, step):
        """Save a checkpoint for the expert model using FSDP state dict utilities"""
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = f"{save_dir}/expert_{self.expert_idx}_step{step}.pt"
            
            # Check if model is wrapped with FSDP
            if isinstance(self.expert, FSDP):
                # Get consolidated state dict using FSDP
                with FSDP.state_dict_type(self.expert, StateDictType.FULL_STATE_DICT):
                    state_dict = self.expert.state_dict()
            else:
                # Regular state dict
                state_dict = self.expert.state_dict()
            
            # Only rank 0 saves the model
            torch.save(state_dict, checkpoint_path)
            return checkpoint_path
        return None
    
    def forward_diffuse(self, x0, t, noise):
        """Forward diffusion process with precomputed alpha_bar"""
        alpha_bar = self.alpha_bar.to(device=self.device, dtype=x0.dtype)
        
        # Extract alpha_bar for the specific timesteps
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
        
        # Apply forward diffusion: x_t = sqrt(α_t)·x_0 + sqrt(1-α_t)·ε
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise 