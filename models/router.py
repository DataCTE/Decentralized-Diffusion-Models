"""Router model implementation (Paper Section 3.3)"""

import torch
import torch.nn as nn
from models.embeddings import TimestepEmbedder

class SelfAttentionBlock(nn.Module):
    """Efficient self-attention block for the router"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ffn_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + attn_out
        
        # FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RouterModel(nn.Module):
    """Implements lightweight router from Paper Section 3.3"""
    def __init__(self, config):
        super().__init__()
        # Paper's temperature schedule
        self.initial_temp = 2.0  # Initial temperature
        self.min_temp = config.router_min_temp
        self.temp_decay = config.router_temperature_decay
        self.current_step = 0
        
        # Embedding layer - Changed input channels to match latent dimension
        self.embedder = nn.Conv2d(
            config.latent_channels,
            config.router_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Efficient spatial attention pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(config.router_hidden_size, config.router_hidden_size // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.router_hidden_size // 4, config.router_hidden_size, kernel_size=1)
        )
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(config.router_hidden_size)
        
        # Text embedding projection - add sequence dimension handling
        self.text_embed_proj = nn.Sequential(
            nn.Linear(config.clip_embedding_dim, config.router_hidden_size),
            nn.ReLU(),
            nn.Linear(config.router_hidden_size, config.router_hidden_size)
        )
        
        # Attention blocks (simplified compared to the DiT)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                config.router_hidden_size,
                config.router_num_heads  # Use direct head count instead of num_heads//2
            )
            for _ in range(2)  # Paper recommends 2 blocks for router
        ])
        
        # Class token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.router_hidden_size))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.router_hidden_size),
            nn.Linear(config.router_hidden_size, config.num_experts)
        )
        
        # Add registered buffer for temperature
        self.temperature = nn.Parameter(torch.tensor(config.router_temperature))
        
        # Initialize weights with smaller values for stability
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedder
        nn.init.xavier_uniform_(self.embedder.weight, gain=0.5)
        nn.init.zeros_(self.embedder.bias)
        
        # Initialize CLS token with small random values
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize final classifier with zeros
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.normal_(self.classifier[-1].weight, std=0.02)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, img, timesteps, txt):
        """Router forward pass with proper argument handling"""
        batch_size = img.shape[0]
        
        # Handle different input dimensions
        if img.dim() == 5:  # [B, C, D, H, W] latent format
            B, C, D, H, W = img.shape
            img = img.reshape(B, C * D, H, W)
        elif img.dim() == 4:  # Standard [B, C, H, W]
            pass
        else:
            raise ValueError(f"Unexpected input dimensions: {img.dim()}")
        
        # Patch embedding
        x = self.embedder(img)  # [B, D, H', W']
        
        # Spatial attention processing
        attn_features = self.spatial_attention(x)
        x = attn_features.mean(dim=(2, 3))  # [B, D]
        
        # Timestep embedding
        t_emb = self.time_embedder(timesteps.float())
        x = x + t_emb

        # Text embedding processing
        if txt.dim() == 4:  # Handle [B, 1, L, D] format
            txt = txt.squeeze(1)
        if txt.dim() == 3:
            txt = self.text_embed_proj(txt).mean(dim=1)
        else:
            txt = self.text_embed_proj(txt)
        
        x = x + txt

        # Transformer processing
        x = x.unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        # Final classification
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        # Apply temperature scaling
        self.temperature.data = torch.max(
            self.temperature * self.temp_decay, 
            torch.tensor(self.min_temp, device=self.temperature.device)
        )
        self.current_step += 1
        
        return logits / self.temperature 