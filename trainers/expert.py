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
        """Train expert with flow matching loss per paper Section 3.2"""
        # Pre-step validation
        if not hasattr(self, 'router') or self.router is None:
            raise RuntimeError(
                f"Expert {self.expert_idx} lost router reference during training! "
                f"Rank {self.rank}, World Size {self.world_size}"
            )
        
        print(f"[DEBUG Expert {self.expert_idx}] Starting train_step")
        print(f"[DEBUG Expert {self.expert_idx}] Batch keys: {batch.keys()}")
        
        # Get data - ensure proper device placement
        latents = batch["latent"].to(self.device, non_blocking=True)
        text_embeds = batch["clip_embedding"].to(self.device, non_blocking=True)
        
        # Get router-predicted clusters with proper timestep scaling
        with torch.no_grad():
            # Generate proper timesteps scaled to [0, 1000)
            router_t = torch.rand(text_embeds.size(0), device=text_embeds.device) * 1000
            
            # Pass scaled timesteps to router
            cluster_ids = self.router.router(
                img=latents,
                txt=text_embeds,
                timesteps=router_t
            ).argmax(dim=-1)
        
        # Print shapes for debugging (using filtered tensors)
        print(f"[DEBUG Expert {self.expert_idx}] Processing {latents.shape[0]} samples for this expert.")
        print(f"[DEBUG Expert {self.expert_idx}] latents shape: {latents.shape}")
        print(f"[DEBUG Expert {self.expert_idx}] text_embeds shape: {text_embeds.shape}")
        print(f"[DEBUG Expert {self.expert_idx}] cluster_ids shape: {cluster_ids.shape}")
        
        # Reshape latents if needed - handle 5D format [B, S, C, H, W] → [B, C, H, W]
        original_shape = latents.shape
        if latents.dim() == 5:
            B, S, C, H, W = latents.shape
            if S == 1:
                latents = latents.squeeze(1)  # Remove sequence dimension if S=1
                print(f"[DEBUG Expert {self.expert_idx}] Reshaped latents to: {latents.shape}")
            else:
                print(f"[WARNING Expert {self.expert_idx}] Multiple sequences ({S}) in batch, using first sequence")
                latents = latents[:, 0]  # Take only first sequence if multiple
                print(f"[DEBUG Expert {self.expert_idx}] Using first sequence, shape: {latents.shape}")
        # --- FIX: Reshape latents to [B, SeqLen, Channels] ---
        B, C, H, W = latents.shape # B is now the number of samples for this expert
        img_seq = latents.reshape(B, C, H * W).permute(0, 2, 1) # Shape: [B_expert, H*W, C]
        print(f"[DEBUG Expert {self.expert_idx}] Reshaped img_seq for model: {img_seq.shape}")
        # --- END FIX ---
        
        # Same for text embeddings if needed
        if text_embeds.dim() == 4:
            B_txt, S_txt, L_txt, D_txt = text_embeds.shape # B_txt is B_expert
            if S_txt == 1:
                text_embeds = text_embeds.squeeze(1)
                print(f"[DEBUG Expert {self.expert_idx}] Reshaped text_embeds to: {text_embeds.shape}")
        
        # Mixed precision context
        with torch.amp.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            # Random timesteps
            t = torch.rand(latents.size(0), device=self.device) # Use B_expert size
            print(f"[DEBUG Expert {self.expert_idx}] timesteps shape: {t.shape}")
            
            try:
                # --- FIX: Use reshaped img_seq and corrected position IDs ---
                # Get H and W from the original latents tensor
                H, W = latents.shape[2], latents.shape[3]  # Use latents instead of undefined xt
                pos_ids = torch.arange(H*W, device=latents.device).repeat(latents.shape[0], 1)
                print(f"[DEBUG Expert {self.expert_idx}] pos_ids shape: {pos_ids.shape}")

                # Forward pass through expert model
                print(f"[DEBUG Expert {self.expert_idx}] Calling expert forward pass")
                pred_flow = self.expert(
                    img=img_seq,                      # Use filtered & reshaped image sequence
                    img_ids=pos_ids,              # Use generated position IDs for filtered batch
                    txt=text_embeds,                  # Use filtered text embeds
                    txt_ids=self._get_text_position_ids(text_embeds), # Use filtered text embeds
                    timesteps=torch.cos(t * math.pi/2), # Use scaled cosine scheduling
                    y=self._get_conditioning(latents.shape[0]), # Use B_expert size
                    cluster_ids=batch['cluster_pred']  # Use router predictions instead of ground truth
                )
                # --- END FIX ---
                print(f"[DEBUG Expert {self.expert_idx}] pred_flow shape: {pred_flow.shape}")
                
                # Calculate loss - need to reshape latents back if it was modified
                print(f"[DEBUG Expert {self.expert_idx}] Computing loss with flow_matcher")
                # Pass the reshaped prediction and original filtered latents to the loss function
                loss = self.flow_matcher.compute_loss(pred_flow, latents, t)
                print(f"[DEBUG Expert {self.expert_idx}] Loss value: {loss.item()}")
                
            except Exception as e:
                print(f"[CRITICAL ERROR Expert {self.expert_idx}] Forward pass failed: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
        
        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        print(f"[DEBUG Expert {self.expert_idx}] Successfully completed train_step")
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
        """Generate position IDs for images with proper dimensionality"""
        # Handle different possible input shapes
        if x.dim() == 5:  # [B, S, C, H, W]
            B, S, C, H, W = x.shape
            if S > 1:
                print(f"[WARNING Expert {self.expert_idx}] Multiple sequences ({S}) in batch for pos IDs, using first.")
            # Use dimensions from the first sequence element if S > 1
            H, W = x.shape[3], x.shape[4]
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Unexpected tensor dimensions for pos IDs: {x.dim()}, shape: {x.shape}")

        # Calculate position ids for a grid
        pos_h = torch.arange(0, H, device=self.device)
        pos_w = torch.arange(0, W, device=self.device)
        
        # Create a grid of coordinates
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Flatten the grid coordinates
        flat_h = grid_h.reshape(-1)
        flat_w = grid_w.reshape(-1)
        
        # Stack to get [H*W, 2] tensor with (y, x) coordinates
        grid_coords = torch.stack([flat_h, flat_w], dim=1)
        
        # Expand to batch dimension [B, H*W, 2]
        pos_ids = grid_coords.unsqueeze(0).expand(B, -1, -1)
        
        print(f"[DEBUG Expert {self.expert_idx}] Position ID output shape: {pos_ids.shape}")
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