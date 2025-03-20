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
    """Implements lightweight router from Paper Appendix A.3"""
    def __init__(self, config):
        super().__init__()
        # Embedding layer
        self.embedder = nn.Conv2d(
            config.latent_channels, 
            config.router_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Efficient spatial attention pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(config.router_hidden_size, config.router_hidden_size // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.router_hidden_size // 4, 1, kernel_size=1)
        )
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(config.router_hidden_size)
        
        # Text embedding projection
        self.text_embed_proj = nn.Linear(config.clip_embedding_dim, config.router_hidden_size)
        
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
        
        # Add temperature decay
        self.temperature = 2.0  # Initial temperature (paper default)
        self.temp_decay = 0.99995  # Paper's annealing rate
        
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

    def forward(self, x, t, text_embeddings):
        """
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B,]
            text_embeddings: Text embeddings for conditioning
        Returns:
            logits: Expert logits [B, num_experts]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.embedder(x)  # [B, D, H', W']
        
        # Spatial attention processing with manual softmax
        attn_features = self.spatial_attention(x)  # [B, 1, H', W']
        attn_flat = attn_features.view(batch_size, 1, -1)
        attn_weights = torch.softmax(attn_flat, dim=2).view_as(attn_features)
        x = (attn_features * attn_weights).sum(dim=(2, 3))  # [B, D]
        x = x.reshape(batch_size, -1) # Ensure x is [B, D] after spatial pooling
        
        # Timestep embedding - ensure float dtype
        t_float = t.float() if t.dtype != torch.float32 else t
        t_float = t_float.to(self.time_embedder.mlp[0].weight.dtype)
        t_emb = self.time_embedder(t_float.unsqueeze(-1))  # [B, D]
        
        # Add timestep information
        x = x + t_emb  # [B, D]
        
        # Add text embedding integration
        text_emb = self.text_embed_proj(text_embeddings)  # Project text embeddings
        x = x + text_emb  # Add projected text embeddings
        
        # Expand to sequence for transformer blocks
        x = x.unsqueeze(1)  # [B, 1, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        print(f"Shape of cls_tokens: {cls_tokens.shape}")
        print(f"Shape of x before cat: {x.shape}")
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 2, D]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Get CLS token output only
        cls_output = x[:, 0]  # [B, D]
        
        # Apply classifier to get logits
        logits = self.classifier(cls_output)  # [B, num_experts]
        
        # Apply temperature scaling for better calibration
        # Lower temperature gives sharper distribution
        return logits / self.temperature
    
    def update_temperature(self):
        """Exponential temperature decay"""
        self.temperature = max(0.5, self.temperature * self.temp_decay)
    
    def get_temperature(self):
        """Get the current temperature value"""
        return self.temperature 