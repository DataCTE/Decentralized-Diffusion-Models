"""Router model implementation for Decentralized Diffusion Models."""

import torch
from torch import nn
import torch.nn.functional as F

from models.embeddings import TimestepEmbedder
from models.dit import DiTBlock

class RouterModel(nn.Module):
    """Router model that predicts which expert to use"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.router_hidden_dim
        self.patch_size = config.patch_size
        self.in_channels = config.latent_channels  # VAE latent channels (16)
        
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
        
        # Transformer blocks (smaller architecture for router)
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, config.num_heads // 2)
            for _ in range(config.num_layers // 4)  # Smaller model for router
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, config.num_experts)
        )
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        
    def get_position_embeddings(self, h, w, device):
        """Generate sinusoidal position embeddings based on grid size"""
        grid_size = (h, w)
        grid_h, grid_w = grid_size
        
        # Generate position embeddings
        pos_embed = torch.zeros(1, grid_h * grid_w, self.hidden_size, device=device)
        grid_indices = torch.cartesian_prod(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device)
        )
        y_indices = grid_indices[:, 0].float()
        x_indices = grid_indices[:, 1].float()
        
        # Calculate embeddings for each dimension
        omega = torch.arange(self.hidden_size // 4, device=device) / (self.hidden_size // 4 - 1)
        omega = 1. / (10000 ** omega)
        
        # Calculate embeddings
        y_embed = y_indices.unsqueeze(1) * omega.unsqueeze(0)
        x_embed = x_indices.unsqueeze(1) * omega.unsqueeze(0)
        
        # Combine embeddings
        pos_embed_raw = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),
            torch.sin(x_embed), torch.cos(x_embed)
        ], dim=1)
        
        # Reshape to correct size
        pos_embed = pos_embed_raw.reshape(1, grid_h * grid_w, self.hidden_size)
        
        return pos_embed
        
    def forward(self, x, t):
        """
        Forward pass through the router model
        
        Args:
            x: Input noisy latent [B, C, H, W]
            t: Timestep [B,]
        """
        # Patchify input
        x = self.x_embedder(x)  # [B, C, H, W] -> [B, D, H/P, W/P]
        
        # Get shapes
        batch_size, _, h, w = x.shape
        
        # Flatten patches
        x = x.flatten(2).permute(0, 2, 1)  # [B, D, H/P, W/P] -> [B, H/P*W/P, D]
        
        # Add position embeddings
        pos_embed = self.get_position_embeddings(h, w, x.device)
        x = x + pos_embed
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Get timestep embeddings
        t_emb = self.t_embedder(t)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Extract CLS token
        cls_token_final = x[:, 0]
        
        # Final layer to predict expert probabilities
        logits = self.final_layer(cls_token_final)  # [B, num_experts]
        
        return logits 