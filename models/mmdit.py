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
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
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
        guidance: Tensor | None = None,
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
        # Dynamic scaling based on sequence length
        B, L_img, _ = img_embed.shape
        H = W = int(math.sqrt(L_img))
        
        # Paper's Equation 8: Resolution-adaptive base frequency
        base_theta = self.params.theta * (H * W / 64)  # Scale for resolution
        
        # Generate grid with adaptive scaling
        y_coords = torch.arange(H, device=img_embed.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=img_embed.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(
            y_coords / H * base_theta,
            x_coords / W * base_theta,
            indexing='ij'
        )
        
        # Combine and flatten
        img_ids = torch.stack([grid_y, grid_x], dim=-1).flatten(0, 1)[None].expand(B, -1, -1)
        return img_ids


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
    num_clusters: int = 8  # Number of data clusters/experts
    cluster_embed_dim: int = 256  # Dimension for cluster embeddings


class ExpertMMDiT(Flux):
    """Implements paper's expert specialization (Section 3.2)"""
    def __init__(self, params: ExpertMMDiTParams):
        # Ensure params has correct attributes before parent init
        self._validate_params(params)
        super().__init__(params)
        
        # Initialize cluster-specific positional embedding
        self.pe_embedder_cluster = EmbedND(
            dim=self.cluster_embed_dim,
            theta=params.theta,
            axes_dim=self.cluster_axes_dim
        )
        
        # Projection for cluster embeddings
        self.cluster_proj = nn.Linear(self.cluster_embed_dim, params.hidden_size)
        
        # Initialize weights properly
        nn.init.normal_(self.cluster_proj.weight, std=0.02)
        nn.init.zeros_(self.cluster_proj.bias)
        
        # Debug flag
        self.debug = True
        
        # Enable gradient checkpointing for large models
        self.double_blocks = nn.ModuleList([
            torch.utils.checkpoint.checkpoint_wrapper(
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                ),
                preserve_rng_state=False
            ) if params.gradient_checkpointing else
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])
        
        # Add gating mechanism for cluster specialization
        self.cluster_gate = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size * 4),
            nn.GELU(),
            nn.Linear(params.hidden_size * 4, params.hidden_size),
            nn.Sigmoid()
        )
        
        # Initialize gate weights properly
        nn.init.orthogonal_(self.cluster_gate[0].weight)
        nn.init.zeros_(self.cluster_gate[0].bias)
        nn.init.orthogonal_(self.cluster_gate[2].weight)
        nn.init.zeros_(self.cluster_gate[2].bias)
        
    def _validate_params(self, params):
        """Validate and prepare parameters for the expert model"""
        # Ensure cluster embed dimension exists and is even (for RoPE)
        self.cluster_embed_dim = getattr(params, 'cluster_embed_dim', 256)
        if self.cluster_embed_dim % 2 != 0:
            self.cluster_embed_dim = self.cluster_embed_dim + 1
            
        # Create a single dimension axes_dim for cluster ID embedding
        self.cluster_axes_dim = [self.cluster_embed_dim]
        
        return params

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        cluster_ids: Tensor,  # [B] cluster indices per sample
        guidance: Tensor | None = None,
    ) -> Tensor:
        if self.debug:
            print(f"[DEBUG MMDiT] img shape: {img.shape}")
            print(f"[DEBUG MMDiT] img_ids shape: {img_ids.shape}")
            print(f"[DEBUG MMDiT] txt shape: {txt.shape}")
            print(f"[DEBUG MMDiT] txt_ids shape: {txt_ids.shape}")
            print(f"[DEBUG MMDiT] timesteps shape: {timesteps.shape}")
            print(f"[DEBUG MMDiT] y shape: {y.shape}")
            print(f"[DEBUG MMDiT] cluster_ids shape: {cluster_ids.shape}")
        
        try:
            # Handle different input shapes from CLIP embeddings
            if txt.ndim == 4 and txt.shape[1] == 1:  # [B, 1, S, D] format
                txt = txt.squeeze(1)
            
            # First, ensure the input tensors have the right dimensions
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError(f"Input img {img.shape} and txt {txt.shape} must have 3 dimensions.")
            
            # Process cluster embeddings
            if cluster_ids.dim() == 1:
                # Add dimension for positional embedding (convert from [B] to [B, 1])
                cluster_ids = cluster_ids.unsqueeze(-1)
            
            # Process cluster embeddings with robust error handling
            try:
                # Generate RoPE embeddings for the cluster IDs
                cluster_embeddings = self.pe_embedder_cluster(cluster_ids)
                if self.debug:
                    print(f"[DEBUG MMDiT] cluster_embeddings shape: {cluster_embeddings.shape}")
                
                # Process embeddings based on dimensionality
                if cluster_embeddings.dim() > 3:
                    # Flatten extra dimensions by averaging
                    cluster_embeddings = cluster_embeddings.mean(dim=-2)
                elif cluster_embeddings.dim() == 3:
                    # If we have a batch x sequence x features tensor, take the mean across sequence
                    cluster_embeddings = cluster_embeddings.mean(dim=1)
                
                # Project to match hidden dimension
                cluster_cond = self.cluster_proj(cluster_embeddings)
                if self.debug:
                    print(f"[DEBUG MMDiT] cluster_cond shape: {cluster_cond.shape}")
                
                # Combine with text condition
                combined_cond = y + cluster_cond
                if self.debug:
                    print(f"[DEBUG MMDiT] combined_cond shape: {combined_cond.shape}")
            except Exception as e:
                print(f"[WARNING] Cluster embedding failed: {e}")
                combined_cond = y  # Fall back to original conditioning
            
            # Process inputs through the transformer architecture
            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256))
            
            if self.params.guidance_embed and guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
                
            # Add conditioning to the vector embedding
            vec = vec + self.vector_in(combined_cond)
            txt = self.txt_in(txt)
            
            # Process positional embeddings
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
            
            # Process through transformer blocks
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
                
            img = torch.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe)
                
            img = img[:, txt.shape[1]:, ...]
            img = self.final_layer(img, vec)
            
            # After combining cluster conditioning
            cluster_emb = self.pe_embedder_cluster(cluster_ids.float()/self.params.num_clusters)
            cluster_proj = self.cluster_proj(cluster_emb)
            
            # Modulate hidden states with cluster-specific gating
            hidden = self.img_in(img) * (1 + self.cluster_gate(cluster_proj))
            
            return hidden
            
        except Exception as e:
            print(f"[CRITICAL ERROR MMDiT] Forward pass failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fall back to parent class implementation as a last resort
            return super().forward(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=y,
                cluster_ids=cluster_ids,
                guidance=guidance
            )

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
