"""Router model implementation (Paper Section 3.3)"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

class RouterModel(nn.Module):
    """Implements lightweight router from Paper Appendix A.3"""
    def __init__(self, config):
        super().__init__()
        self.embedder = nn.Conv2d(config.latent_channels, config.router_hidden_dim,
                                kernel_size=config.patch_size,
                                stride=config.patch_size)
        
        self.blocks = nn.ModuleList([
            FSDP(nn.TransformerEncoderLayer(
                config.router_hidden_dim, 
                config.num_heads//2,
                dim_feedforward=config.router_hidden_dim*4,
                batch_first=True
            ), sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
            for _ in range(2)
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.router_hidden_dim))
        self.classifier = nn.Linear(config.router_hidden_dim, config.num_experts)

    def forward(self, x, t):
        x = self.embedder(x).flatten(2).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.blocks:
            x = block(x)
            
        return self.classifier(x[:, 0]) 