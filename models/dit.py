"""DiT implementation for Decentralized Diffusion Models (Paper Section 3.2)"""

import torch
import torch.nn as nn
import math

from models.embeddings import TimestepEmbedder

def modulate(x, shift, scale):
    """Applies modulation to the input"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """Implements transformer block with adaLN-zero conditioning (Paper Eq. 4)"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulated attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa)
        )[0]
        
        # Modulated MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

class FinalLayer(nn.Module):
    """Final layer with adaLN modulation"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        
        # Initialize to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        # Get AdaLN modulation parameters
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply modulation
        x = modulate(self.norm_final(x), shift, scale)
        
        # Project to output
        return self.linear(x)

class ExpertDiT(nn.Module):
    """Implements expert model from Paper Section 3.2"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.patch_size = config.patch_size
        self.in_channels = config.latent_channels  # VAE latent channels (16)
        self.out_channels = config.latent_channels  # Predict latent noise
        
        # Embeddings
        self.x_embedder = nn.Conv2d(
            self.in_channels, 
            self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Position embedding
        self.pos_embed = None  # Will be generated dynamically
        
        # CLIP text conditioning components
        self.text_projection = nn.Linear(768, self.hidden_size)  # Project CLIP embeddings to DiT hidden dim
        self.text_cross_attention = nn.MultiheadAttention(
            self.hidden_size, 
            config.num_heads, 
            batch_first=True
        )
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, config.num_heads, config.ffn_dim/config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        
        # Initialize weights
        self.initialize_weights()
        
        # Enable gradient checkpointing for memory efficiency from config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
    def initialize_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize modulation layers to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
    def get_position_embeddings(self, h, w, device):
        """Generate position embeddings for arbitrary grid sizes"""
        grid_indices = torch.stack(torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        ), dim=-1).float()  # [H, W, 2]
        
        # Calculate embeddings for each dimension
        omega = torch.arange(self.hidden_size // 4, device=device) / (self.hidden_size // 4 - 1)
        omega = 1. / (10000 ** omega)
        
        # Calculate embeddings
        y_embed = grid_indices[..., 0:1] * omega
        x_embed = grid_indices[..., 1:2] * omega
        
        # Combine embeddings and flatten
        pos_embed = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),
            torch.sin(x_embed), torch.cos(x_embed)
        ], dim=-1).flatten(0, 1).unsqueeze(0)  # [1, H*W, D]
        
        return pos_embed
            
    def unpatchify(self, x, h, w):
        """Reshape patches back to image with explicit h/w dimensions"""
        batch_size = x.shape[0]
        
        # Reshape to [B, H, W, patch_size, patch_size, C]
        x = x.reshape(batch_size, h, w, self.patch_size, self.patch_size, self.out_channels)
        
        # Permute and reshape to image
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, self.out_channels, 
                     h * self.patch_size, w * self.patch_size)
        return x
    
    def forward(self, x, t, text_embeds=None):
        # FSDP-compatible forward pass
        x = self.x_embedder(x)  # [B, C, H, W] -> [B, D, H/P, W/P]
        x = x.flatten(2).permute(0, 2, 1)  # [B, D, N] -> [B, N, D]
        
        # Generate dynamic position embeddings
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        self.pos_embed = self.get_position_embeddings(h, w, x.device)
        
        # Add position embeddings with sharding awareness
        x += self.pos_embed.to(x.dtype)  # FSDP handles sharded parameters
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, t)
            else:
                x = block(x, t)
        
        # Final projection
        return self.final_layer(x, t) 