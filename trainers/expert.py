"""Expert trainer for Decentralized Diffusion Models."""

import torch
from bitsandbytes.optim import AdamW8bit
import math
import os
import torch.nn.functional as F
from utils.distributed import is_dist_initialized, synchronize
from utils.fsdp import wrap_model_with_fsdp, configure_optimizer_for_fsdp

from models.mmdit import ExpertMMDiT
from trainers.diffusion import DecentralizedFlowMatcher, get_alphas_and_betas
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder
from trainers.base import BaseTrainer
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint
from utils.logging import logger

# Import FSDP explicitly for type checking
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP



class ExpertTrainer(BaseTrainer):
    """
    Trainer for expert DiT models in DDM.
    Each expert trains in complete isolation on its assigned data cluster,
    with no cross-communication between experts as described in paper Section 3.2
    """
    def __init__(self, expert_idx, config, device, rank, world_size, router=None):
        # Paper-recommended initialization (section 4.1)
        super().__init__(config, device, rank)
        
        # Validate router existence before initializing components
        if router is None:
            raise ValueError(
                f"ExpertTrainer requires router reference. "
                f"Missing router for expert {expert_idx} (rank {rank})"
            )
        
        self.router = router
        self.expert_idx = expert_idx
        self.world_size = world_size
        
        # --- FIX: Ensure model config uses latent_channels for in_channels ---
        from types import SimpleNamespace
        model_config_dict = vars(config).copy() # Create a mutable copy
        if hasattr(config, 'latent_channels'):
             model_config_dict['in_channels'] = config.latent_channels
             logger.info(f"Expert {expert_idx}: Setting model in_channels to latent_channels ({config.latent_channels})")
        else:
             logger.warning(f"Expert {expert_idx}: config.latent_channels not found, using config.in_channels ({config.in_channels}) for model.")
        # Use a SimpleNamespace or a dedicated dataclass if ExpertMMDiT expects one
        model_config = SimpleNamespace(**model_config_dict)
        # --- END FIX ---

        # Initialize base model first using the corrected config
        self.expert = ExpertMMDiT(model_config) # Pass the modified config
        
        # Don't wrap with FSDP here - let cache manager handle it
        
        # Use standard optimizer initially - will be reconfigured when FSDP is applied
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
        self.vae = None
        self.clip = None
        
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

        # Initialize scaler once in __init__
        self.scaler = torch.amp.GradScaler(enabled=config.use_mixed_precision)

    def compute_loss(self, batch):
        """Paper's per-expert loss calculation (Equation 6), aligned with train_step logic"""
        # FSDP handles device placement - no .to(device) needed
        x0 = batch["latent"]  # Already on correct device

        # --- Handle potential 5D input from dataloader ---
        if x0.dim() == 5:
            B, S, C, H, W = x0.shape
            if S == 1:
                x0 = x0.squeeze(1)
            else:
                # Use first sequence if multiple exist
                x0 = x0[:, 0]
        # --- End 5D handling ---

        # Sample timesteps uniformly as in paper
        t = torch.rand(x0.size(0), device=x0.device)  # Use tensor's device

        # Forward process using paper's flow matching formulation
        # Ensure alpha_t and sigma_t match the dimensions of x0 (4D)
        alpha_t = torch.cos(t * math.pi/2).view(-1, 1, 1, 1)
        sigma_t = torch.sin(t * math.pi/2).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = alpha_t * x0 + sigma_t * noise # xt is 4D: [B, C, H, W]

        # --- Reshape xt for the model ---
        B, C, H, W = xt.shape
        xt_seq = xt.reshape(B, C, H * W).permute(0, 2, 1) # Shape: [B, H*W, C]
        # --- End reshape ---

        # --- Prepare text embeddings (handle potential 4D) ---
        text_embeds = batch['clip_embedding']
        if text_embeds.dim() == 4:
            B_txt, S_txt, L_txt, D_txt = text_embeds.shape
            if S_txt == 1:
                text_embeds = text_embeds.squeeze(1)
            else:
                text_embeds = text_embeds[:, 0] # Use first sequence
        # --- End text embedding prep ---

        # Expert forward pass with cluster conditioning
        # Use reshaped xt_seq and scaled timesteps
        img_pos_ids = self._get_position_ids(xt) # Pass original 4D xt to get H, W
        pred_flow = self.expert(
            img=xt_seq,                       # Use reshaped image sequence
            img_ids=img_pos_ids,              # Use generated position IDs
            txt=text_embeds,
            txt_ids=self._get_text_position_ids(text_embeds),
            timesteps=t * 1000,               # Scale timesteps like in train_step
            y=self._get_conditioning(x0.shape[0]),
            cluster_ids=batch['cluster_pred']  # Use router predictions instead of ground truth
        )

        # --- Use flow_matcher for loss calculation ---
        # Pass the model's prediction (pred_flow) and the original data (x0)
        loss = self.flow_matcher.compute_loss(pred_flow, x0, t)
        # --- End flow_matcher usage ---

        return loss # flow_matcher.compute_loss already returns a scalar loss item

    def train_step(self, batch):
        # Paper-recommended modifications
        self.expert.train()
        self.router.eval()  # Freeze router during expert training
        
        # Get router predictions with temperature annealing
        with torch.no_grad():
            router_t = torch.rand(batch["latent"].size(0), device=self.device) * 1000
            cluster_probs = F.softmax(
                self.router(batch["latent"], router_t, batch["clip_embedding"]) / 
                max(self.config.router_min_temp, 
                    self.config.router_temperature * (self.config.router_temperature_decay ** self.step)),
                dim=-1
            )
            cluster_ids = torch.multinomial(cluster_probs, 1).squeeze(-1)

        # Filter batch for expert's cluster
        expert_mask = (cluster_ids == self.expert_idx)
        if not torch.any(expert_mask):
            return None  # No samples for this expert
        
        expert_batch = {
            k: v[expert_mask] for k,v in batch.items() 
            if not isinstance(v, list)
        }

        # Add capacity-aware filtering
        expert_capacity = int(self.config.expert_batch_size * self.config.expert_capacity_factor)
        
        # After getting expert_mask
        if expert_mask.sum() > expert_capacity:
            # Paper's capacity-aware random selection
            selected_indices = torch.randperm(expert_mask.sum(), device=self.device)[:expert_capacity]
            expert_mask[expert_mask.nonzero()[selected_indices]] = False

        # Paper's recommended forward pass
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            # Convert latent to sequence format
            B, C, H, W = expert_batch["latent"].shape
            img_seq = expert_batch["latent"].view(B, C, H*W).permute(0, 2, 1)
            
            # Get position IDs
            pos_ids = self._get_position_ids(expert_batch["latent"])
            
            pred_flow = self.expert(
                img=img_seq,
                img_ids=pos_ids,
                txt=expert_batch["clip_embedding"],
                txt_ids=self._get_text_position_ids(expert_batch["clip_embedding"]),
                timesteps=torch.rand(B, device=self.device) * 1000,
                y=self._get_conditioning(B),
                cluster_ids=cluster_ids[expert_mask]
            )
            
            # Reshape prediction to match latent
            pred_flow = pred_flow.permute(0, 2, 1).view(B, C, H, W)
            
            # Paper's equation 6
            loss = self.flow_matcher.compute_loss(
                pred_flow, 
                expert_batch["latent"],
                torch.rand(B, device=self.device)
            )

        # Gradient handling per paper appendix
        self.scaler.scale(loss).backward()
        if self.step % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.expert.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
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
    
    def forward_diffuse(self, x0, t, noise=None):
        """
        Forward diffusion process with paper's cosine schedule
        Section 3.1 and Appendix A
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Paper's cosine schedule
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        
        # Forward process: xt = αt·x0 + σt·ε
        return alpha_t * x0 + sigma_t * noise

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

    def _get_position_ids(self, x):
        # Get spatial dimensions
        if x.dim() == 4:
            _, _, H, W = x.shape
        else:  # Handle sequence format
            H = W = int(math.sqrt(x.shape[1]))
        
        # Generate grid with matching dtype and device
        device = x.device
        dtype = x.dtype
        pos_h = torch.arange(H, device=device, dtype=dtype)
        pos_w = torch.arange(W, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Stack and flatten to [B, H*W, 2]
        pos_ids = torch.stack([grid_h, grid_w], dim=-1).flatten(0, 1)[None]  # [1, L, 2]
        pos_ids = pos_ids.expand(x.shape[0], -1, -1)  # [B, L, 2]
        
        return pos_ids

    def _get_text_position_ids(self, text_emb):
        """Simple sequence position IDs for text with improved shape handling"""
        print(f"[DEBUG Expert {self.expert_idx}] Text embedding shape for position IDs: {text_emb.shape}")
        
        if text_emb.dim() == 4:
            # Handle [B, S, L, D] → [B, L, D]
            B, S, L, D = text_emb.shape
            if S == 1:
                # If sequence dimension is 1, we can simply remove it
                print(f"[DEBUG Expert {self.expert_idx}] Reshaping 4D text embedding for position IDs")
                text_emb = text_emb.squeeze(1)
            else:
                print(f"[WARNING Expert {self.expert_idx}] Multiple text sequences ({S}), using first")
                text_emb = text_emb[:, 0]
        
        # Now process the standard 3D case
        B, L, _ = text_emb.shape
        pos_ids = torch.arange(L, device=self.device)[None].repeat(B, 1)
        pos_ids = pos_ids[:, :, None].repeat(1, 1, 2)  # Add 2D position dim for consistency
        
        print(f"[DEBUG Expert {self.expert_idx}] Text position ID shape: {pos_ids.shape}")
        return pos_ids

    def _get_conditioning(self, batch_size):
        """Get conditioning vector for expert"""
        # Return zero conditioning vector of correct shape
        return torch.zeros(batch_size, self.config.vec_in_dim, device=self.device) 