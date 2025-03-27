"""
MMDiT model implementation for Decentralized Diffusion Models.
This file integrates modular components from the modules directory.
"""

import torch
from torch import Tensor, nn
from dataclasses import dataclass
import math


# Import core modules
from models.modules.layers import (
    EmbedND, 
    MLPEmbedder, 
    DoubleStreamBlock, 
    SingleStreamBlock, 
    LastLayer,
    timestep_embedding
)
from models.modules.lora import LinearLora, replace_linear_with_lora

# Import config
from config import get_config

# Get active config (will load defaults if not initialized elsewhere)
config = get_config()


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    latent_channels: int
    gradient_checkpointing: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.latent_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if len(params.axes_dim) != 2:  # Require 2D position encoding
            raise ValueError(f"Positional axes must be 2D for image data")
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Axes dim {params.axes_dim} sum != {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                torch.utils.checkpoint.checkpoint(
                    DoubleStreamBlock(
                        self.hidden_size,
                        self.num_heads,
                        mlp_ratio=params.mlp_ratio,
                        qkv_bias=params.qkv_bias,
                    )
                ) if params.gradient_checkpointing else
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        cluster_ids: Tensor | None = None,
    ) -> Tensor:
        # Paper's cluster conditioning (Equation 5)
        if cluster_ids is not None:
            # Convert cluster IDs to embeddings
            cluster_emb = self.pe_embedder_cluster(
                cluster_ids.float() / self.params.num_clusters
            )
            # Project to hidden dimension
            cluster_proj = self.cluster_proj(cluster_emb)
            # Add to text conditioning vector
            y = y + cluster_proj[:, None]  # [B, 1, D] -> [B, S, D]
        
        # Unified dtype handling
        dtype = next(self.parameters()).dtype
        img, txt = img.to(dtype), txt.to(dtype)
        
        # Fused embedding projections
        img_emb = self.img_in(img) * (1 + self.time_in(timesteps)[:, None])
        txt_emb = self.txt_in(txt) + self.vector_in(y)[:, None]
        
        # Combine embeddings
        x = torch.cat([txt_emb, img_emb], dim=1)
        pos_ids = torch.cat([txt_ids, img_ids], dim=1)
        
        # Process through transformer blocks
        for block in self.double_blocks:
            x = block(x, vec=timesteps, pe=self.pe_embedder(pos_ids))
        
        return self.final_layer(x[:, txt_emb.size(1):], timesteps)

    def generate_position_ids(self, txt_embed: Tensor, img_embed: Tensor):
        # Dynamic scaling based on input resolution
        B, L_img, _ = img_embed.shape
        H = W = int(math.sqrt(L_img))
        
        # Base frequency scaling (paper's Eq.8)
        base_theta = self.params.theta * (H * W / 64)  # 64 base resolution
        
        # Generate grid coordinates
        y_coords = torch.linspace(0, base_theta, H, device=img_embed.device)
        x_coords = torch.linspace(0, base_theta, W, device=img_embed.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Combine spatial coordinates
        img_ids = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        return img_ids.unsqueeze(0).expand(B, -1, -1)


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)

@dataclass 
class ExpertMMDiTParams(FluxParams):
    """Parameters for expert MMDiT aligned with config defaults"""
    # Cluster/expert configuration
    num_clusters: int = config.num_clusters
    cluster_embed_dim: int = config.cluster_embed_dim
    expert_capacity_factor: float = config.expert_capacity_factor
    
    # Architecture parameters from config
    hidden_size: int = config.hidden_size
    depth: int = config.depth
    num_heads: int = config.num_heads
    mlp_ratio: float = config.mlp_ratio
    qkv_bias: bool = config.qkv_bias
    
    # Positional embedding configuration
    theta: int = config.theta
    position_embed_type: str = config.position_embedding
    
    # Performance configurations
    gradient_checkpointing: bool = config.use_gradient_checkpointing

class ExpertMMDiT(Flux):
    """Implements paper's expert specialization (Section 3.2) with capacity awareness"""
    def __init__(self, params: ExpertMMDiTParams):
        # Remove guidance-related components from base params
        params.guidance_embed = False  # Disable unused guidance embedding
        
        self._validate_params(params)
        super().__init__(params)

        # Paper's cluster embedding initialization (Section 4.1)
        self.cluster_embed = nn.Sequential(
            nn.Embedding(params.num_clusters, params.cluster_embed_dim),
            nn.Linear(params.cluster_embed_dim, params.hidden_size)
        )
        
        # Initialize with scaled normal distribution per paper
        nn.init.normal_(self.cluster_embed[0].weight, std=0.02/math.sqrt(params.hidden_size))
        nn.init.zeros_(self.cluster_embed[1].weight)
        nn.init.zeros_(self.cluster_embed[1].bias)

        # Paper's capacity-aware initialization (Section 3.4)
        self.register_buffer('expert_capacity', 
            torch.tensor(params.expert_capacity_factor, dtype=torch.float32))
            
    def _validate_params(self, params):
        """Ensure cluster config matches paper specifications"""
        if not hasattr(params, 'num_clusters'):
            raise ValueError("ExpertMMDiT requires num_clusters parameter")
        if params.cluster_embed_dim % 2 != 0:
            params.cluster_embed_dim += 1  # Ensure even dim for RoPE
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("Hidden size must be divisible by num_heads")

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Validate cluster IDs (Section 3.4)
        if (cluster_ids < 0).any() or (cluster_ids >= self.params.num_clusters).any():
            invalid = cluster_ids.unique().tolist()
            raise ValueError(f"Invalid cluster IDs {invalid}. Must be in [0, {self.params.num_clusters-1}]")
        
        # Cluster conditioning (Equation 5)
        cluster_emb = self.cluster_embed(cluster_ids)  # [B, D]
        
        # Capacity-aware scaling (Section 3.4)
        capacity_scale = 1.0 + self.expert_capacity * torch.sigmoid(cluster_emb.mean(dim=-1))
        conditioned_y = y * capacity_scale[:, None] + cluster_emb.unsqueeze(1)
        
        # Original processing with capacity conditioning
        return super().forward(img, img_ids, txt, txt_ids, timesteps, conditioned_y, cluster_ids)

    def create_embeddings(self, text_input, image_input):
        """
        Create embeddings from text and image inputs
        
        Args:
            text_input: Either text prompts (list of strings) or text embeddings
            image_input: Either image tensors or image embeddings
            
        Returns:
            Dictionary with 'txt', 'txt_ids', 'img', 'img_ids'
        """
        # Handle text input
        if isinstance(text_input, list) and isinstance(text_input[0], str):
            # This is a list of text prompts
            if hasattr(self, 'text_embedder_type') and self.text_embedder_type == 't5':
                # Use T5 embedder if specified
                if not hasattr(self, 'text_embedder'):
                    from models.modules.conditioner import T5Embedder
                    self.text_embedder = T5Embedder("google/t5-v1_1-base", max_length=128).to(self.device)
                txt_embed = self.text_embedder(text_input)
            else:
                # Default to CLIP embedder
                if not hasattr(self, 'text_embedder'):
                    from models.modules.conditioner import CLIPEmbedder
                    self.text_embedder = CLIPEmbedder("openai/clip-vit-large-patch14", max_length=77).to(self.device)
                txt_embed = self.text_embedder(text_input)
        else:
            # Assume these are already embeddings
            txt_embed = text_input
        
        # Handle image input
        if isinstance(image_input, torch.Tensor) and image_input.dim() == 4:
            # Convert [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = image_input.shape
            img_embed = image_input.reshape(B, C, H*W).permute(0, 2, 1)
        else:
            # Assume these are already embeddings
            img_embed = image_input
        
        # Generate position IDs
        txt_ids, img_ids = self.generate_position_ids(txt_embed, img_embed)
        
        return {
            'txt': txt_embed,
            'txt_ids': txt_ids,
            'img': img_embed,
            'img_ids': img_ids
        }

    def generate(self, prompts, num_inference_steps=50, guidance_scale=7.5, latents=None, callback=None):
        """
        Generate images from text prompts
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            latents: Optional initial latents
            callback: Optional callback function for progress
            
        Returns:
            Generated image tensors
        """
        # Import necessary diffusion modules
        from trainers.diffusion import get_alphas_and_betas, ddim_step
        
        # Create text embeddings with classifier-free guidance
        if not hasattr(self, 'text_embedder'):
            from models.modules.conditioner import CLIPEmbedder
            self.text_embedder = CLIPEmbedder("openai/clip-vit-large-patch14", max_length=77)
            self.text_embedder = self.text_embedder.to(next(self.parameters()).device)
        
        # Get text embeddings
        text_embeddings, uncond_embeddings = self.text_embeddings.encode_with_uncond(prompts)
        
        # Prepare for classifier-free guidance
        batch_size = text_embeddings.shape[0]
        
        # Create initial latents if not provided
        if latents is None:
            shape = (batch_size, 4, 64, 64)  # Default latent shape
            latents = torch.randn(shape, device=self.device)
        
        # Setup diffusion parameters
        alphas, alpha_bar, _ = get_alphas_and_betas(num_inference_steps)
        alphas = alphas.to(self.device)
        alpha_bar = alpha_bar.to(self.device)
        
        # Create a simple progress tracker
        from tqdm.auto import tqdm
        progress_bar = tqdm(range(num_inference_steps))
        
        # Generate dummy cluster IDs (or use actual ones if available)
        cluster_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Sample loop
        for t in progress_bar:
            # Create timestep tensor
            timesteps = torch.full((batch_size,), t, device=self.device)
            
            # Get positional embeddings
            txt_len = text_embeddings.shape[1]
            txt_ids = torch.arange(txt_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
            img_shape = latents.shape
            img_len = img_shape[2] * img_shape[3]  # H*W of latents
            img_ids = torch.arange(img_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
            
            # Add dimension for RoPE
            txt_ids = txt_ids.unsqueeze(-1)
            img_ids = img_ids.unsqueeze(-1)
            
            # Reshape image for model input
            B, C, H, W = latents.shape
            img_input = latents.reshape(B, C, H*W).permute(0, 2, 1)
            
            # Classifier-free guidance
            noise_pred_uncond = self(
                img=img_input,
                img_ids=img_ids,
                txt=uncond_embeddings,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=uncond_embeddings.mean(dim=1),
                cluster_ids=cluster_ids
            )
            
            noise_pred_text = self(
                img=img_input,
                img_ids=img_ids,
                txt=text_embeddings,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=text_embeddings.mean(dim=1),
                cluster_ids=cluster_ids
            )
            
            # Combine with guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Get next timestep
            t_next = torch.full((batch_size,), t+1, device=self.device) if t < num_inference_steps-1 else None
            
            # Perform DDIM step
            latents = ddim_step(
                lambda x_t, t, c: noise_pred,
                latents,
                timesteps,
                t_next,
                alphas,
                alpha_bar,
                eta=0.0
            )
            
            # Call the callback if provided
            if callback is not None:
                callback(latents, t)
        
        # Return final latents
        return latents

    def deterministic_sample(self, noise: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Implements paper's deterministic sampler (Appendix B)"""
        alpha_bar = torch.cos(torch.linspace(0, 1, steps+1) * math.pi/2)
        for t in reversed(range(steps)):
            # Paper's modified reverse process
            pred = self(noise, t)
            noise = (noise - (1 - alpha_bar[t])*pred) / alpha_bar[t]
        return noise
