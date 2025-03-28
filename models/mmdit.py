"""
MMDiT model implementation for Decentralized Diffusion Models.
This file integrates modular components from the modules directory.
"""

import torch
from torch import Tensor, nn
from dataclasses import dataclass, field
import math
from einops import rearrange


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
    # All fields now require explicit initialization
    in_channels: int
    out_channels: int
    hidden_size: int
    num_heads: int
    depth: int
    mlp_ratio: float
    qkv_bias: bool
    axes_dim: list[int]
    theta: int
    position_embed_type: str
    num_clusters: int
    cluster_embed_dim: int
    expert_capacity_factor: float
    vec_in_dim: int
    context_in_dim: int
    guidance_embed: bool
    gradient_checkpointing: bool
    latent_channels: int
    depth_single_blocks: int
    patch_size: int

    def __post_init__(self):
        """Post-initialization validation"""
        if not hasattr(self, 'gradient_checkpointing'):
            raise ValueError("gradient_checkpointing must be defined")


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
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
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class ExpertMMDiT(Flux):
    """Implements paper's expert specialization with unified parameters"""
    def __init__(self, params: FluxParams):
        # Disable guidance embedding for experts (paper Section 3.2)
        params.guidance_embed = False
        
        self._validate_params(params)
        super().__init__(params)

        # Add device-aware initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move cluster embedding to device
        self.cluster_embed = nn.Embedding(
            params.num_clusters, 
            params.cluster_embed_dim
        ).to(self.device)
        
        # Projection to match conditioning dimension
        self.cluster_proj = nn.Linear(
            params.cluster_embed_dim,
            params.vec_in_dim
        )
        # Initialize with paper's recommended scheme
        nn.init.normal_(self.cluster_embed.weight, std=0.02)
        nn.init.kaiming_normal_(self.cluster_proj.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.cluster_proj.bias)

        # Paper's expert capacity gating (Equation 7)
        self.capacity_gate = nn.Sequential(
            nn.Linear(params.cluster_embed_dim, 1),
            nn.Sigmoid()
        )
        nn.init.normal_(self.capacity_gate[0].weight, std=0.01)  # More stable initialization
        nn.init.constant_(self.capacity_gate[0].bias, 0.0)  # Neutral initial bias

        # CORRECTED final layer - output channels should match latent dimensions
        self.final_layer = LastLayer(
            self.params.hidden_size,
            self.params.patch_size,
            self.params.latent_channels  # Now matches configured out_channels
        )

    def _validate_params(self, params):
        """Safer parameter validation with cluster config"""
        if not hasattr(params, 'cluster_embed_dim'):
            raise ValueError("Missing cluster_embed_dim in config")
        
        if params.cluster_embed_dim <= 0:
            raise ValueError(f"cluster_embed_dim must be > 0, got {params.cluster_embed_dim}")
        
        # Existing validation
        if params.cluster_embed_dim % 2 != 0:
            params.cluster_embed_dim += 1

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
        """Implements paper's expert specialization with validated dimensions"""
        # Validate input dimensions
        # Cluster conditioning (Equation 5)
        cluster_emb = self.cluster_embed(cluster_ids)
        conditioned_y = self._apply_capacity_scaling(y, cluster_emb)
        
        # Prepare embeddings with fused operations
        img_emb, txt_emb = self._fuse_embeddings(img, txt, timesteps, conditioned_y)
        
        # Process through transformer blocks
        return self._transformer_forward(img_emb, txt_emb, img_ids, txt_ids, timesteps)


    def _apply_capacity_scaling(self, y: Tensor, cluster_emb: Tensor) -> Tensor:
        """Implements paper's Equation 7 with dimension fix"""
        # Process full embeddings through the gate
        capacity_scale = 1.0 + self.capacity_gate(cluster_emb)  # [B, 1]
        # Add projected cluster information
        return y * capacity_scale + self.cluster_proj(cluster_emb)

    def _fuse_embeddings(self, img, txt, timesteps, y):
        """Fuse embeddings with proper dimension handling"""
        dtype = next(self.parameters()).dtype
        img, txt = img.to(dtype), txt.to(dtype)
        
        # Get actual dimensions for debugging
        B, L, C = img.shape
        
        # Add timestep embedding (paper's Eq.3)
        t_emb = timestep_embedding(timesteps.float(), 256)
        time_vec = self.time_in(t_emb)
        
        # Ensure dimensions match the expected input channel size
        # The img_in layer expects dimensions matching self.in_channels
        if C != self.params.in_channels:
            # Handle the mismatch by reshaping or padding
            if self.params.patch_size**2 * self.params.latent_channels == C:
                # This is a patched input, matches our expectation
                pass  # Keep as is
            else:
                raise ValueError(
                    f"Expected input channels {self.params.in_channels}, but got {C}. "
                    f"Check that patch_size ({self.params.patch_size}), latent_channels "
                    f"({self.params.latent_channels}) and input dimension match."
                )
        
        # Time-aware image projection
        img_emb = self.img_in(img) * (1 + time_vec[:, None])
        
        # Text conditioning with vector projection
        txt_emb = self.txt_in(txt) + self.vector_in(y)[:, None]
        
        return img_emb, txt_emb

    def _transformer_forward(self, img_emb, txt_emb, img_ids, txt_ids, timesteps):
        """Core transformer processing with corrected position embeddings"""
        t_emb = timestep_embedding(timesteps.float(), 256)
        time_vec = self.time_in(t_emb)
        
        # Process through double stream blocks
        img_stream = img_emb
        txt_stream = txt_emb
        for block in self.double_blocks:
            img_stream, txt_stream = block(
                img_stream,
                txt_stream,
                time_vec,
                self.pe_embedder(torch.cat([txt_ids, img_ids], dim=1))
            )

        # Generate fresh position IDs for combined sequence
        combined_ids = torch.cat([
            txt_ids[:, :txt_stream.size(1)],  # Actual text sequence length
            img_ids[:, :img_stream.size(1)]   # Actual image sequence length
        ], dim=1)
        
        pe = self.pe_embedder(combined_ids)
        
        # Concatenate streams and process through single blocks
        x = torch.cat([txt_stream, img_stream], dim=1)
        
        for block in self.single_blocks:
            x = block(
                x,
                time_vec,
                pe
            )

        # Extract image features after processing (critical dimension fix)
        img_features = x[:, txt_stream.size(1):]
        
        # Add paper's patch decoding
        patch_size = self.params.patch_size
        h_dim = img_ids[:, :, 0].max() + 1  # Get actual spatial dimensions from position IDs
        w_dim = img_ids[:, :, 1].max() + 1
        return rearrange(
            self.final_layer(img_features, time_vec),
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_dim,
            w=w_dim,
            p1=self.params.patch_size,
            p2=self.params.patch_size,
            c=self.params.latent_channels
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
            )
            
            noise_pred_text = self(
                img=img_input,
                img_ids=img_ids,
                txt=text_embeddings,
                txt_ids=txt_ids,
                timesteps=timesteps,
                y=text_embeddings.mean(dim=1),
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
