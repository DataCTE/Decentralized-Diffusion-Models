from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from PIL import Image
import os
from functools import lru_cache
from transformers import AutoModel, AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, MixedPrecision
from dataclasses import dataclass
import time
import datetime
import math
from torch.quantization import quantize_dynamic 
from timm.models.vision_transformer import PatchEmbed, Mlp
from tqdm import tqdm
from adamw_bf16 import AdamWBF16
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

@dataclass
class ModelConfig:
    hidden_dim: int = 1152  # DiT-XL dimension
    num_layers: int = 28     # DiT-XL depth
    num_heads: int = 16      # DiT-XL heads
    ffn_dim: int = 3072      # DiT-XL MLP ratio 4x
    num_experts: int = 8
    learning_rate: float = 1e-4
    adam_betas: tuple = (0.9, 0.99)
    weight_decay: float = 0.1
    patch_size: int = 32
    image_size: int = None
    out_channels: int = 16
    depth: int = 12
    capacity_factor: float = 1.0  # Paper sec 3.1
    drop_tokens: bool = True      # Paper sec 3.1
    aux_loss_weight: float = 0.1  # Paper eq 5
    utilization_weight: float = 0.01  # λ from paper sec 3.4
    num_steps: int = 1_000_000  # Total training steps
    log_dir: str = "runs/main"

def init_dinov2():
    """Initialize DINOv2 without xformers modifications"""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
    
    # Split transformer blocks across 7 GPUs
    num_blocks = len(model.blocks)
    blocks_per_gpu = math.ceil(num_blocks / 7)
    for i, block in enumerate(model.blocks):
        device_idx = i // blocks_per_gpu
        block.to(f'cuda:{device_idx}')
    
    # Distribute remaining components
    model.patch_embed = model.patch_embed.to('cuda:0')
    model.norm = model.norm.to(f'cuda:{min(6, num_blocks//blocks_per_gpu)}')
    model.head = model.head.to('cuda:6')
    
    # Freeze parameters and set eval mode
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    
    return model


def setup_distributed():
    # Get ranks from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set device before any CUDA calls
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group with explicit device_id
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=3600),  # Increased from 300 to 3600
        device_id=device  # Explicit device assignment
    )
    
    # Verify device assignment
    current_device = torch.cuda.current_device()
    assert current_device == local_rank, \
        f"Device mismatch: {current_device} vs {local_rank}"
    
    # Warmup NCCL (critical for multi-node)
    torch.distributed.all_reduce(torch.zeros(1).to(device))
    
    return rank, local_rank, world_size

#################################################################################
#                                   modulate                                    #
#################################################################################

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).to(x.dtype)) + shift.unsqueeze(1).to(x.dtype)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, 
                end=half, 
                dtype=torch.float32,  # Keep math in float32
                device=t.device
            ) / half
        )
        args = t[:, None].to(torch.float32) * freqs[None]  # Maintain float32 for precision
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding  # Return float32 for MLP processing

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))  # Explicit cast before linear layers
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings.to(torch.bfloat16)  # Cast to match model precision


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    Official DiT block with adaLN-Zero conditioning
    """
    def __init__(self, hidden_size, num_heads, num_experts=8, mlp_ratio=4.0, config=None, **block_kwargs):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.register_buffer(
            "scale", 
            torch.sqrt(torch.tensor(hidden_size // num_heads)),
            persistent=False
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Official initialization from MAE
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.constant_(self.attn.bias, 0.)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # MoDE specific components
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Mlp(
                in_features=hidden_size, 
                hidden_features=mlp_hidden_dim,
                act_layer=nn.GELU
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_size, num_experts)
        self.gate = nn.Softmax(dim=-1)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention path
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # Use standard PyTorch attention
        attn_out, _ = self.attn(
            attn_input, 
            attn_input,
            attn_input,
            need_weights=False
        )
        
        # Apply gating to attention output
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP path
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        
        # MoDE routing with capacity factor (paper sec 3.1)
        router_logits = self.router(x)  # [B, N, E]
        routing_weights = self.gate(router_logits.mean(1))  # [B, E]
        
        # Calculate capacity with safety clamp
        capacity = int(x.size(1) * self.config.capacity_factor)
        capacity = max(1, min(capacity, self.num_experts))  # Ensure 1 <= capacity <= num_experts
        
        expert_assign = torch.topk(routing_weights, k=capacity, dim=1).indices
        
        # Create mask using one-hot encoding
        expert_mask = torch.nn.functional.one_hot(expert_assign, num_classes=self.num_experts)
        expert_mask = expert_mask.sum(dim=1)  # Shape: [B, num_experts]
        
        # Process through experts with token dropping
        x = x.unsqueeze(2).expand(-1, -1, self.num_experts, -1)  # [B, N, E, D]
        expert_outputs = torch.stack([e(x[:, :, i]) for i, e in enumerate(self.experts)], dim=2)  # [B, N, E, D]
        
        # Combine using mask and average remaining tokens
        x = (expert_outputs * expert_mask.unsqueeze(-1).unsqueeze(1)).sum(dim=2)  # [B, N, D]
        norm_factor = expert_mask.sum(dim=1).view(-1, 1, 1)  # Reshape to [B, 1, 1]
        x = x / (norm_factor + 1e-6)  # Normalize

        return x, router_logits

class FinalLayer(nn.Module):
    """Official DiT final layer with learned adaLN parameters"""
    def __init__(self, hidden_size, patch_size, out_channels, num_experts):
        super().__init__()
        self.num_experts = num_experts
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Router for expert selection
        self.mode_fc = nn.Linear(hidden_size, num_experts * 2)
        
        # Build a list of experts – ensure the number matches the router's expectation.
        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, patch_size**2 * out_channels)
            for _ in range(num_experts)
        ])
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # Apply learned modulation
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x_mod = modulate(self.norm_final(x), shift, scale)  # x_mod: [B, N, D]
        
        # Compute each expert's prediction independently.
        expert_outputs = torch.stack([expert(x_mod) for expert in self.experts], dim=2)  # [B, N, num_experts, out_dim]
        
        # Compute router logits from the token mean.
        router_logits = self.mode_fc(x_mod.mean(dim=1))  # [B, num_experts * 2]
        routing_weights = router_logits[..., :self.num_experts]  # [B, num_experts]
        
        # Select the top expert (k=1) for each sample.
        expert_assign = torch.topk(routing_weights, k=1, dim=-1).indices  # [B, 1]
        expert_mask = torch.nn.functional.one_hot(expert_assign, num_classes=self.num_experts)  # [B, 1, num_experts]
        expert_mask = expert_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_experts, 1]
        
        # Combine the expert outputs using the mask.
        output = (expert_outputs * expert_mask).sum(dim=2)  # [B, N, out_dim]
        return output, router_logits

class DiT(nn.Module):
    """Official DiT implementation with exact parameter initialization"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.learn_sigma = True  # Follow paper default
        self.in_channels = 16  # Matches 16-channel VAE
        self.out_channels = 16 * 2 if self.learn_sigma else 16
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads

        self.x_embedder = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.t_embedder = TimestepEmbedder(config.hidden_dim)
        self.y_embedder = LabelEmbedder(1000, config.hidden_dim, 0.1)
        self.pos_embed = None  # Will be generated dynamically

        # Initialize transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                config.hidden_dim, 
                config.num_heads, 
                num_experts=config.num_experts,
                mlp_ratio=config.ffn_dim/config.hidden_dim,
                config=config
            ) for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.hidden_dim, config.patch_size, self.out_channels, config.num_experts)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, t, y):
        # Dynamic patch embedding
        x = self.x_embedder(x)  # [B, D, H, W]
        B, D, H, W = x.shape  # Get spatial dimensions from patch embedder
        # Store the original grid dimensions from the patch embedder
        self._grid_size = (H, W)
        # Generate positional embeddings for HxW grid
        pos_embed = get_2d_sincos_pos_embed(D, grid_h=H, grid_w=W, device=x.device)
        x = x.flatten(2).transpose(1, 2) + pos_embed  # [B, N, D]
        
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        
        for block in self.blocks:
            x, router_logits = block(x, c)
            
        x, router_logits = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x, router_logits

    def unpatchify(self, x):
        # If x does not have exactly three dimensions (B, N, patch_dim),
        # flatten all dimensions except the batch and last dimension.
        if x.dim() != 3:
            B = x.size(0)
            patch_dim = x.size(-1)
            # Compute token count N by multiplying all intermediate dimensions.
            N = 1
            for dim in x.shape[1:-1]:
                N *= dim
            x = x.view(B, N, patch_dim)
        B, N, patch_dim = x.shape  # patch_dim should equal p*p*out_channels
        p = self.x_embedder.kernel_size[0]  # patch size
        out_channels = patch_dim // (p * p)
        
        # Retrieve the original grid dimensions from the patch embedder.
        orig_h, orig_w = self._grid_size
        expected_tokens = orig_h * orig_w

        # If token count N does not match the expected grid,
        # try to infer h and w by factorizing N.
        if N != expected_tokens:
            import math
            h_candidate = int(math.sqrt(N))
            while h_candidate > 1 and N % h_candidate != 0:
                h_candidate -= 1
            w_candidate = N // h_candidate
            h, w = h_candidate, w_candidate
        else:
            h, w = orig_h, orig_w

        # Reshape the tokens into patches and reconstruct the image.
        x = x.reshape(B, h, w, p, p, out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, out_channels, h * p, w * p)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device, cls_token=False, extra_tokens=0):
    grid_h = torch.arange(grid_h, device=device, dtype=torch.float32)  # Keep as float32
    grid_w = torch.arange(grid_w, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)
    grid = grid.reshape(2, 1, grid_h.shape[0], grid_w.shape[0])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros(extra_tokens, embed_dim, device=device, dtype=pos_embed.dtype), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return torch.cat([emb_h, emb_w], dim=1)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)  # Match input dtype
    omega /= embed_dim / 2.
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

#################################################################################
#                               Diffusion Process                               #
#################################################################################

def get_alphas_and_betas(num_timesteps=1000, schedule_type='cosine'):
    """Compute noise schedule coefficients (α, α_bar, β) for DDPM"""
    if schedule_type == 'cosine':
        # Improved DDPM cosine schedule (paper sec. 3.2)
        max_beta = 0.999
        ts = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((ts / num_timesteps + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(max_beta))
    else:  # linear schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    
    alphas = 1. - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar, betas

def forward_diffuse(x0, t, noise, alpha_bar):
    """
    Diffuse data through time using precomputed coefficients
    Args:
        x0: Original images [B, C, H, W]
        t: Timestep indices [B,] 
        noise: Pre-generated noise [B, C, H, W]
        alpha_bar: Precomputed cumulative product of alphas [T,]
    Returns:
        x_t: Noised version of x0 at timestep t [B, C, H, W]
    """
    alpha_bar = alpha_bar.to(device=x0.device, dtype=x0.dtype)
    
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1. - alpha_bar[t])[:, None, None, None]
    
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

#################################################################################

def save_checkpoint(step):
    torch.save({
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'step': step
    }, f'checkpoints/dit_step_{step}.pth')

def load_checkpoint(step):
    checkpoint = torch.load(f'checkpoints/dit_step_{step}.pth')
    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])

#################################################################################
#                              Unified Implementation                           #
#################################################################################

class UnifiedDataset(Dataset):
    def __init__(self, root_dir, buckets, dit_patch_size=32, dino_patch_size=14, local_rank=0, rank=0):
        self.dit_patch_size = dit_patch_size
        self.dino_patch_size = dino_patch_size
        
        # Dual validation for both model requirements
        self.buckets = [
            (self._round_to_patch(w, dit_patch_size), 
             self._round_to_patch(h, dit_patch_size))
            for w, h in buckets
        ]
        
        self.root = root_dir
        self.local_rank = local_rank
        self.rank = rank
        self.cluster_version = 0
        
        # Set device FIRST before any CUDA operations
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)  # Ensure proper device context
        
        # Initialize cluster management with device context
        self.cluster_mgr = ClusterManager(local_rank)

        # Ensure all buckets are multiples of patch size
        self.buckets = [
            (self._round_to_patch(w, dit_patch_size), self._round_to_patch(h, dit_patch_size))
            for w, h in buckets
        ]
        # Add validation for buckets
        for bw, bh in self.buckets:
            assert bw % dit_patch_size == 0 and bh % dit_patch_size == 0, \
                f"Invalid bucket size ({bw},{bh}) for patch size {dit_patch_size}"

        # Initialize samples FIRST
        self.samples = self._validate_files()
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {root_dir}")
            
        print(f"[Rank {self.rank}] Found {len(self.samples)} valid samples")
        
        # Add bucket grouping AFTER samples initialization
        self.bucket_groups = defaultdict(list)
        for idx in range(len(self.samples)):
            bucket = self._find_bucket(*Image.open(self.samples[idx]['image']).size)
            self.bucket_groups[bucket].append(idx)
            
        # Validate bucket groups
        if not self.bucket_groups:
            raise RuntimeError("No valid buckets created - check input image sizes")
            
        logger.info(f"Created {len(self.bucket_groups)} bucket groups")
        
        # Bucket-aware transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
        # THEN initialize cluster labels
        self.cluster_labels = np.zeros(len(self.samples), dtype=np.int64)  # Move this AFTER samples init
        
        # Distributed cluster precompute with detailed logging
        if self.rank == 0:
            logger.info("Starting latent feature precomputation for clustering")
            
        max_retries = 3
        fallback_used = False
        for attempt in range(max_retries):
            try:
                cluster_tensor = torch.zeros(len(self.samples), dtype=torch.long, device=self.device)
                if self.rank == 0:
                    total_features = 0
                    start_time = time.time()
                    features = []
                    
                    logger.info(f"Processing {len(self.bucket_groups)} bucket groups")
                    for bucket_idx, (bucket, indices) in enumerate(self.bucket_groups.items()):
                        logger.info(f"Processing bucket {bucket} with {len(indices)} samples")
                        bucket_start = time.time()
                        
                        bucket_features = self._safe_feature_extraction(bucket, indices)
                        if bucket_features is not None:
                            features.append(bucket_features)
                            total_features += len(indices)
                            logger.info(f"Bucket {bucket} completed in {time.time()-bucket_start:.2f}s - "
                                      f"Features extracted: {len(indices)}")
                        else:
                            logger.warning(f"Bucket {bucket} returned no features")

                    if features:
                        all_features = torch.cat(features)
                        logger.info(f"Clustering {total_features} total features across {len(features)} buckets")
                        cluster_start = time.time()
                        cluster_labels = self.cluster_mgr.cluster_features(all_features.numpy())
                        logger.info(f"Clustering completed in {time.time()-cluster_start:.2f}s")
                        cluster_tensor = torch.tensor(cluster_labels, device=self.device)
                
                # Broadcast with logging
                if self.rank == 0:
                    logger.info("Initiating cluster label broadcast to all nodes")
                work = torch.distributed.broadcast(cluster_tensor, src=0, async_op=True)
                if self._wait_for_distributed_op(work, 10800):
                    logger.info("Cluster label broadcast completed successfully")
                else:
                    logger.error("Cluster label broadcast timed out after 3 hours")
                
                self.cluster_labels = cluster_tensor.cpu().numpy()
                break
            
            except Exception as e:
                logger.error(f"Cluster precompute attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries-1:
                    logger.warning("Using fallback cluster labels after 3 failures")
                    fallback_used = True
                    fallback_tensor = torch.zeros(len(self.samples), dtype=torch.long, device=self.device)
                    work = torch.distributed.broadcast(fallback_tensor, src=0, async_op=True)
                    if self._wait_for_distributed_op(work, 3600):
                        logger.info("Fallback cluster broadcast completed")
                    else:
                        logger.error("Fallback cluster broadcast timed out after 1 hour")
                    self.cluster_labels = fallback_tensor.cpu().numpy()

        # Final validation logging
        if self.rank == 0:
            logger.info(f"Cluster initialization completed with {len(np.unique(self.cluster_labels))} clusters")
            if fallback_used:
                logger.warning("Used fallback cluster labels due to initialization failures")
            logger.info(f"Total cluster initialization time: {time.time()-start_time:.2f} seconds")

    def _round_to_patch(self, dimension, patch_size):
        """Force dimensions to be multiples of patch size using ceiling division"""
        return patch_size * ((dimension + patch_size - 1) // patch_size)

    def _validate_files(self):
        """Properly handles image/text file pairs"""
        valid_samples = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        # First pass: find all potential image files
        all_files = os.listdir(self.root)
        image_candidates = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        # Second pass: validate image/text pairs
        for img_file in tqdm(image_candidates, desc="Validating pairs"):
            try:
                base_name = os.path.splitext(img_file)[0]
                txt_file = f"{base_name}.txt"
                txt_path = os.path.join(self.root, txt_file)
                
                # Check text file exists
                if not os.path.exists(txt_path):
                    logger.debug(f"Missing text file for {img_file}")
                    continue
                    
                # Validate image file
                img_path = os.path.join(self.root, img_file)
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                    
                valid_samples.append({
                    'image': img_path,
                    'text': txt_path
                })
                
            except Exception as e:
                logger.warning(f"Invalid pair {img_file}: {str(e)}")
                continue
                
        logger.info(f"Found {len(valid_samples)} valid image/text pairs")
        return valid_samples

    def __getitem__(self, idx):
        try:
            filename = self.samples[idx]
            img_path = filename['image']
            txt_path = filename['text']
            
            # Load image with enhanced error handling
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if min(img.size) < 16:  # Prevent tiny images
                    raise ValueError(f"Image too small: {img.size}")
                
                # Enforce strict bucket sizing with padding
                w, h = img.size
                target_w, target_h = self._find_bucket(w, h)
                
                # Calculate padding to maintain aspect ratio
                scale = min(target_w/w, target_h/h)
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)
                
                img = img.resize((scaled_w, scaled_h), Image.BICUBIC)
                
                # Modified padding logic:
                if scaled_w != target_w or scaled_h != target_h:
                    # Ensure we don't create invalid dimensions
                    pad_w = max(target_w - scaled_w, 0)
                    pad_h = max(target_h - scaled_h, 0)
                    img = TF.pad(img, [pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2])
                
                # Final size validation
                final_w, final_h = img.size
                assert final_w % self.dit_patch_size == 0 and final_h % self.dit_patch_size == 0, \
                    f"Processed size {final_w}x{final_h} not multiple of {self.dit_patch_size}"
                
                img_tensor = self.transform(img)  # Now only does ToTensor + Normalize
                if torch.isnan(img_tensor).any():
                    raise ValueError("NaN values in image tensor")
                    
            # Load text with encoding validation
            with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read().strip()
                if not text:
                    raise ValueError("Empty text file")
            
            # Get cluster label
            cluster = self.cluster_labels[idx]
            
            # Then process separate copy for DINOv2
            dino_transform = T.Resize((self._round_to_patch(img.height, 14),
                                     self._round_to_patch(img.width, 14)))
            dino_img = dino_transform(img)
            return {
                'dit_image': img_tensor,
                'text': text,
                'cluster': torch.tensor(cluster, dtype=torch.long),
                'bucket': (target_w, target_h),
                'dino_image': dino_img
            }
            
        except Exception as e:
            logger.warning(f"Skipping sample {idx}: {str(e)}")
            return None

    def __len__(self):
        """Required for DataLoader to determine dataset size"""
        return len(self.samples)

    def _extract_features(self):
        all_features = []
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        loader = DataLoader(
            self,
            batch_size=128,  # Reduced from 256
            num_workers=2,    # Reduced from 4
            collate_fn=lambda x: [item for item in x if item['dit_image'] is not None]
        )
        
        for batch in loader:
            try:
                if not batch:
                    continue
                    
                images = torch.stack([item['dit_image'] for item in batch]).to(self.device)
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    features = self.cluster_mgr.dino(images)
                    all_features.append(features.cpu())
                    
            except Exception as e:
                continue

        return torch.cat(all_features) if all_features else None

    def _safe_feature_extraction(self, bucket, indices):
        """Distributed feature extraction with memory mapping"""
        # Create memory-mapped array
        features_shape = (len(indices), 1024)
        features = np.memmap(f'/dev/shm/features_{self.rank}', 
                           dtype=np.float32, mode='w+', shape=features_shape)
        
        # Split work across 7 GPUs
        indices_per_rank = np.array_split(indices, 7)
        my_indices = indices_per_rank[self.rank]
        
        # Pre-warm GPUs with synthetic data
        if not hasattr(self, '_gpu_warmed'):
            self._warm_gpu()
            self._gpu_warmed = True
        
        # Process local chunk
        for batch_idx in range(0, len(my_indices), 512):
            batch_indices = my_indices[batch_idx:batch_idx+512]
            batch = [self.__getitem__(i) for i in batch_indices]
            
            # Async pipeline
            images = torch.stack([item['dit_image'] for item in batch]).pin_memory()
            images = images.to(f'cuda:{self.local_rank}', non_blocking=True)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
                features = self.cluster_mgr.dino(images).float()
            
            # Direct memory mapping
            features = features.cpu().numpy()
            features[batch_idx:batch_idx+len(batch)] = features
        
        # Synchronize across nodes
        torch.distributed.barrier()
        return features

    def _warm_gpu(self):
        """Pre-warm GPU with synthetic batches"""
        warm_data = torch.randn(32, 3, 256, 256, device=f'cuda:{self.local_rank}')
        for _ in range(3):
            _ = self.cluster_mgr.dino(warm_data)
        torch.cuda.synchronize()

    def _wait_for_distributed_op(self, work, timeout_sec):
        """Handles distributed operation timeouts"""
        start = time.time()
        while not work.is_completed():
            if time.time() - start > timeout_sec:
                return False
            time.sleep(1)
        return True

    def _find_bucket(self, w, h):
        """Updated bucket matching with proper rounding"""
        # Use rounded dimensions for matching
        w_round = self._round_to_patch(w, self.dit_patch_size)
        h_round = self._round_to_patch(h, self.dit_patch_size)
        
        # Find closest bucket with aspect ratio priority
        aspect = w_round / h_round
        candidates = []
        for bw, bh in self.buckets:
            if bw % self.dit_patch_size != 0 or bh % self.dit_patch_size != 0:
                continue  # Skip invalid buckets
            bucket_aspect = bw / bh
            aspect_diff = abs(aspect - bucket_aspect)
            size_diff = abs(bw - w_round) + abs(bh - h_round)
            candidates.append((aspect_diff, size_diff, (bw, bh)))
        
        # Prioritize aspect ratio match, then size proximity
        if candidates:
            return min(candidates)[2]
        
        # Fallback: use smallest valid bucket
        valid_buckets = [b for b in self.buckets 
                        if b[0]%self.dit_patch_size==0 and b[1]%self.dit_patch_size==0]
        return min(valid_buckets, key=lambda x: x[0]*x[1])

class ClusterManager:
    """Implements paper's cluster management with online updates"""
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Initialize components
        self.dino = self._init_vision_encoder()
        self.text_encoder, self.tokenizer = self._init_text_encoder()
        
        # Cluster components
        self.coarse_clusterer = MiniBatchKMeans(n_clusters=8)
        self.fine_clusterer = MiniBatchKMeans(n_clusters=1024)

    def _init_vision_encoder(self):
        """Parallel DINOv2 initialization across 7 GPUs"""
        # Load base model with explicit patch size
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)
        self.dino_patch_size = 14  # Critical for feature extraction
        
        # Add DINO-specific preprocessing
        self.dino_preprocess = T.Compose([
            T.Resize((self.dino_patch_size * 14, self.dino_patch_size * 14)),
            T.CenterCrop((self.dino_patch_size * 14, self.dino_patch_size * 14)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Split model across 7 GPUs using pipeline parallelism
        num_blocks = len(model.blocks)
        blocks_per_gpu = math.ceil(num_blocks / 7)
        
        # Distribute blocks across devices
        for i, block in enumerate(model.blocks):
            device_idx = i // blocks_per_gpu
            block.to(f'cuda:{device_idx}')
        
        # Distribute remaining components
        model.patch_embed.to('cuda:0')
        model.norm.to(f'cuda:{min(6, num_blocks//blocks_per_gpu)}')
        model.head.to('cuda:6')
        
        # Freeze parameters and set eval mode
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        
        return model

    def _init_text_encoder(self):
        """Paper's SimCSE initialization with quantization (sec 4.2)"""
        tokenizer = AutoTokenizer.from_pretrained(
            "princeton-nlp/sup-simcse-bert-base-uncased",
            padding_side='right',
            use_fast=True,
            model_max_length=77
        )
        
        model = AutoModel.from_pretrained(
            "princeton-nlp/sup-simcse-bert-base-uncased",
            trust_remote_code=True
        )
        
        state_dict = torch.hub.load_state_dict_from_url(
            "https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased/resolve/main/pytorch_model.bin",
            map_location='cpu',
            progress=False
        )
        
        state_dict = {k: v for k,v in state_dict.items() if "position_ids" not in k}
        model.load_state_dict(state_dict, strict=False)
        
        quantized_model = quantize_dynamic(
            model.to(self.device),
            {nn.Linear},
            dtype=torch.qint8
        )
        
        return quantized_model.eval(), tokenizer

    def cluster_features(self, features):
        """Faiss GPU-accelerated clustering"""
        import faiss
        
        # Configure for 7 GPUs
        res = [faiss.StandardGpuResources() for _ in range(7)]
        index = faiss.IndexFlatL2(features.shape[1])
        
        # Shard across GPUs
        index = faiss.index_cpu_to_gpu_multiple_py(res, index, [0,1,2,3,4,5,6])
        
        # Train and cluster
        index.train(features)
        cluster_labels = index.assign(features)[1]
        return cluster_labels

class UnifiedSampler(torch.utils.data.Sampler):
    """Implements paper's cluster-balanced sampling (sec 4.3)"""
    def __init__(self, groups, batch_size, num_replicas, rank):
        self.groups = groups
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        cluster_counts = np.array([len(v) for v in self.groups.values()])
        weights = 1 / np.sqrt(cluster_counts)
        probs = weights / weights.sum()
        
        sampled_clusters = np.random.choice(
            list(self.groups.keys()),
            size=len(self.groups),
            p=probs,
            replace=True
        )
        
        all_batches = []
        for c in sampled_clusters:
            indices = np.random.permutation(self.groups[c])
            all_batches.extend(np.array_split(indices, len(indices)//self.batch_size))
        
        np.random.shuffle(all_batches)
        per_replica = len(all_batches) // self.num_replicas
        return iter(np.concatenate(all_batches[self.rank*per_replica:(self.rank+1)*per_replica]))

def collate_fn(batch):
    """Handles paper's multi-resolution batches (sec 4.1)"""
    # Filter out None entries and invalid images
    valid_batch = [b for b in batch if b is not None and b['dit_image'] is not None]
    
    if not valid_batch:
        return {'dit_image': torch.empty(0), 'text': [], 'cluster': torch.empty(0)}
    
    # Group by bucket dimensions
    bucket_groups = defaultdict(list)
    for item in valid_batch:
        if 'bucket' not in item:
            continue  # Skip items without bucket info
        bucket_key = tuple(item['bucket']) if isinstance(item['bucket'], (list, tuple)) else item['bucket']
        bucket_groups[bucket_key].append(item)
    
    # Process each bucket group
    batched = []
    for (w, h), group in bucket_groups.items():
        if not group:
            continue
            
        # Stack images and collect metadata
        images = []
        texts = []
        clusters = []
        for item in group:
            if item['dit_image'] is not None:
                images.append(item['dit_image'])
                texts.append(item['text'])
                clusters.append(item['cluster'])
        
        if not images:
            continue
            
        batched.append({
            'dit_image': torch.stack(images),
            'text': texts,
            'cluster': torch.stack(clusters) if clusters else None,
            'bucket': (w, h)
        })
    
    return batched



# --- Training System ---
class MoDETrainer:
    def __init__(self, config: ModelConfig, dataset=None, rank=None, local_rank=None, world_size=None):
        self.rank = rank
        print(f"[Rank {self.rank}] Initializing trainer...")
        
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Set device first
        self.device = torch.device(f"cuda:{local_rank}")

        # Initialize TensorBoard logging (minimal overhead)
        self.tb_writer = SummaryWriter(log_dir=config.log_dir)

        # Distributed data loader
        self.loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            ),
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        # Model setup - convert to BF16
        self.model = DiT(config).to(dtype=torch.bfloat16).to(self.device)
        self.model = DDP(
            self.model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Allow parameters that are not used every forward pass
        )
        
        self.optimizer = ZeroRedundancyOptimizer(
            self.model.parameters(),
            optimizer_class=AdamWBF16,
            lr=config.learning_rate,
            betas=config.adam_betas,
            weight_decay=config.weight_decay,
            overlap_with_ddp=False
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_steps
        )

        self.step = 0  # Step counter

        self.empty_cache_interval = 10
        self.last_cache_empty = 0

        self.max_keep_models = 1  # Keep only 1 model in memory at a time
        self.current_model = None

        import atexit
        atexit.register(self._cleanup_resources)

        self._shared_models = {
            'vae': None,
            'dino': None,
            'simcse': None
        }

        logger.info(f"Initializing MoDE Trainer on GPU {local_rank}")
        logger.debug(f"World size: {world_size}, Batch size: {BATCH_SIZE}")
        logger.info(f"Model config: {config}")

        if rank == 0:
            print(f"\n{'='*40}")
            print(f" Distributed Training Startup ")
            print(f"World Size: {world_size}")
            print(f"Batch Size: {BATCH_SIZE * world_size}")
            print(f"Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Using GPUs: {[f'cuda:{i}' for i in range(world_size)]}")
            print(f"{'='*40}\n")

        self.alphas, self.alpha_bar, self.betas = get_alphas_and_betas()
        self.alpha_bar = self.alpha_bar.to(self.device)
        self.num_timesteps = len(self.betas)

        self.ema_update_interval = 10
        self.router_ema = self._create_ema()

        print(f"[Rank {self.rank}] Trainer initialized. Dataset length: {len(dataset)}")

    def _create_ema(self):
        """Paper's EMA implementation (sec 5.2)"""
        class EMA:
            def __init__(self, model, decay=0.9999):
                self.model = model  # Store model reference
                self.decay = decay
                self.shadow = {}
                self.original = {}
                
                # Register parameters
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
                        self.original[name] = param.data.clone()

            def update(self):
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.shadow[name] = (
                                self.decay * self.shadow[name] + 
                                (1 - self.decay) * param.data
                            )

            def apply(self):
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self.original[name] = param.data.clone()
                            param.data.copy_(self.shadow[name])

            def restore(self):
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param.data.copy_(self.original[name])
                            
        return EMA(self.model)

    @property
    def vae(self):
        if self._shared_models['vae'] is None:
            self._shared_models['vae'] = AutoencoderKL.from_pretrained(
                "AuraDiffusion/16ch-vae"
            ).to(dtype=torch.bfloat16, device=self.device)
            self._shared_models['vae'].eval()
            for param in self._shared_models['vae'].parameters():
                param.requires_grad_(False)
        return self._shared_models['vae']
    
    @property
    def dino(self):
        if self._shared_models['dino'] is None:
            self._shared_models['dino'] = init_dinov2()
        return self._shared_models['dino']
    
    
    @property
    def simcse(self):
        """Access through EnhancedClusterManager (paper sec 4.2)"""
        return self.cluster_mgr.text_encoder

    def _release_models(self):
        """Release all shared models from memory"""
        logger.debug("Releasing shared models from memory")
        for key in self._shared_models:
            if self._shared_models[key] is not None:
                del self._shared_models[key]
                self._shared_models[key] = None
        torch.cuda.empty_cache()

    def train_batch(self, batch):
        if self.rank == 0:
            print(f"\n[Rank 0] Starting training step {self.step}")
        total_loss = 0
        start_time = time.perf_counter()
        
        valid_batches = [b for b in batch if len(b['dit_image']) > 0]
        if not valid_batches:
            return 0.0
            
        accum_steps = len(valid_batches)
        self.optimizer.zero_grad()
        
        for resolution_batch in valid_batches:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                images = resolution_batch['dit_image'].to(
                    self.device, non_blocking=True, memory_format=torch.channels_last
                )
                clusters = resolution_batch['cluster'].to(self.device, non_blocking=True)
                images = images * 2 - 1

                with torch.inference_mode():
                    latents = self.vae.encode(images).latent_dist.sample()
                    latents = latents * 0.18215

            t = torch.randint(0, self.num_timesteps, (images.size(0),), device=self.device, dtype=torch.long)
            noise = torch.randn_like(latents, dtype=torch.bfloat16)
            x_t = forward_diffuse(latents, t, noise, self.alpha_bar)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred, router_logits = torch.utils.checkpoint.checkpoint(
                    self.model, x_t, t, clusters,
                    use_reentrant=False,
                    preserve_rng_state=True
                )

                if hasattr(self.model, "module"):
                    model_ref = self.model.module
                else:
                    model_ref = self.model

                if model_ref.learn_sigma:
                    noise_pred = pred[:, :model_ref.in_channels, :, :]
                else:
                    noise_pred = pred

                if noise_pred.shape[-2:] != noise.shape[-2:]:
                    from torchvision.transforms import functional as TF
                    noise_pred = TF.center_crop(noise_pred, noise.shape[-2:])
                    
                mse_loss = F.mse_loss(noise_pred, noise)

            probs = torch.softmax(router_logits, dim=-1)
            aux_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
            expert_mask = torch.sigmoid(router_logits)
            utilization = expert_mask.mean(dim=0)
            load_loss = torch.std(utilization) * self.config.utilization_weight
            loss = mse_loss + self.config.aux_loss_weight * aux_loss + load_loss
            
            (loss / accum_steps).backward()
            total_loss += loss.detach().item()

            if self.step % self.ema_update_interval == 0:
                self.router_ema.update()
                
            del images, latents, noise, x_t, pred
            torch.cuda.empty_cache()

        if any(p.grad is not None for p in self.model.parameters()):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        end_time = time.perf_counter()
        if self.rank == 0 and self.step % 100 == 0:
            print(f"\nStep {self.step} Summary:")
            print(f"Total Loss: {total_loss/accum_steps:.4f}")
            print(f"MSE: {mse_loss.item():.4f} | Aux: {aux_loss.item():.4f} | Load: {load_loss.item():.4f}")
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            print(f"Time: {end_time - start_time:.3f}s")
            
            # TensorBoard logging at regular intervals
            self.tb_writer.add_scalar("Train/TotalLoss", total_loss/accum_steps, global_step=self.step)
            self.tb_writer.add_scalar("Train/MSE", mse_loss.item(), global_step=self.step)
            self.tb_writer.add_scalar("Train/AuxLoss", aux_loss.item(), global_step=self.step)
            self.tb_writer.add_scalar("Train/LoadLoss", load_loss.item(), global_step=self.step)
            self.tb_writer.add_scalar("Train/StepTime", end_time - start_time, global_step=self.step)
            self.tb_writer.flush()  # <-- Ensure logs are flushed to disk

        self.step += 1
        return total_loss / sum(len(b['dit_image']) for b in valid_batches)

    def validate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in test_loader:
                images = batch['dit_image'].to(self.device)
                t = torch.randint(0, self.num_timesteps, (images.size(0),), device=self.device)
                noise = torch.randn_like(images)
                x_t = self.alpha_bar[t] * images + self.betas[t] * noise
                
                pred, _ = self.model(x_t, t)
                total_loss += F.mse_loss(pred, noise).item()
                
        self.model.train()
        return total_loss / len(test_loader)

    def generate(self, prompt, size=(256, 256), num_steps=50):
        """Paper's cluster-conditioned generation (algorithm 2)"""
        self.model.eval()
        
        # Encode text and get cluster
        with torch.no_grad():
            # Text feature extraction (paper sec 4.2)
            text_inputs = self.cluster_mgr.tokenizer(
                prompt,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_feats = self.cluster_mgr.text_encoder(**text_inputs).last_hidden_state.mean(1)
            cluster_id = self.cluster_mgr.online_updater.predict(text_feats.cpu().numpy())
        
        # EMA averaging (paper sec 5.2)
        with self.router_ema.average_parameters(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Paper's generation process
            z = torch.randn((1, 3, *size), device=self.device)  # Initial noise
            
            # Timestep schedule
            timesteps = torch.linspace(0, self.num_timesteps-1, num_steps, device=self.device).long()
            
            for t in reversed(timesteps):
                # Expand timestep for batch dimension
                timestep = torch.full((1,), t, device=self.device, dtype=torch.long)
                
                # Model prediction (conditioned on cluster)
                pred_noise, _ = self.model(z, timestep, cluster=torch.tensor([cluster_id], device=self.device))
                
                # Reverse diffusion step (paper eq 7)
                z = self._previous_step(z, pred_noise, t)
                
            # Decode with VAE
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                return self.vae.decode(z / 0.18215).sample

    def _cleanup_resources(self):
        """Explicitly release multiprocessing resources"""
        if self.loader._iterator is not None:
            self.loader._iterator._shutdown_workers()
        torch.distributed.destroy_process_group()
        self.router_ema = None  # Release EMA buffers

class AverageMeter:
    """Utility class for tracking metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class OptimizedCLIPTextModel(CLIPTextModel):
    def forward(self, input_ids, attention_mask=None):
        # Remove xformers-specific context managers
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

class LazyModule(nn.Module):
    def __init__(self, module_class, *args, **kwargs):
        super().__init__()
        self.module_class = module_class
        self.args = args
        self.kwargs = kwargs
        self._module = None

    def forward(self, *inputs):
        if self._module is None:
            self._module = self.module_class(*self.args, **self.kwargs).to(inputs[0].device)
        return self._module(*inputs)

# --- Hardcoded Training Setup ---
if __name__ == "__main__":
    # Remove xformers warning suppression
    # Keep PyTorch's native memory optimizations
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision('high')
    
    # Configuration (paper table 5)
    DATA_DIR = "/path/to/dataset"
    BATCH_SIZE = 256  # Global batch size
    NUM_STEPS = 100000 # 1M steps for base training
    VAL_INTERVAL = 1000 
    WARMUP_STEPS = 100
    LOG_DIR = "runs/main"
    
    # Initialize distributed
    rank, local_rank, world_size = setup_distributed()
    torch.cuda.set_device(local_rank)
    
    # Paper's model configuration (DiT-XL)
    """default vaules:
    hidden_dim=1152, 
    num_layers=28,     
    num_heads=16,     
    ffn_dim=3072,
    """
    model_config = ModelConfig(
        patch_size=32,  # SDXL-compatible patch size
        hidden_dim=1152, 
        num_layers=28,     
        num_heads=16,     
        ffn_dim=3072,
        image_size=1024  # Max training resolution
    )
    
    # Dataset with paper's bucketing
    buckets = [
        (512, 512),    # 32x32 latent (16x downsampling)
        (768, 768),    # 24x24 latent
        (1024, 1024),  # 32x32 latent
        (640, 1536),   # 20x48 latent
        (1536, 640),   # 48x20 latent
        (896, 1152),   # 28x36 latent
        (1152, 896)    # 36x28 latent
    ]
    base_dataset = UnifiedDataset(
        DATA_DIR, 
        buckets,
        dit_patch_size=32,
        dino_patch_size=14,
        local_rank=local_rank,
        rank=rank
    )
        
    # Trainer with paper's settings
    trainer = MoDETrainer(
        config=model_config,
        dataset=base_dataset,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size
    )
    
    # Paper's learning rate schedule
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, NUM_STEPS - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    # Training loop (paper algorithm 1)
    try:
        for step in range(1, NUM_STEPS+1):
            # Train step
            batch = next(iter(trainer.loader))
            loss = trainer.train_batch(batch)
            
            # Validation and logging
            if step % VAL_INTERVAL == 0:
                val_loss = trainer.validate(trainer.loader)
                
                # Paper's logging format
                if rank == 0:
                    print(f"\nStep {step}")
                    print(f"Train Loss: {loss:.4f}")
                    print(f"Val Loss: {val_loss:.4f}")
                    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                    
                    # Save checkpoint
                    save_checkpoint(step)
                    
    except KeyboardInterrupt:
        save_checkpoint(step)
        print(f"\nSaved final checkpoint at step {step}")

    if rank == 0:
        print("\nInitialization Complete:")
        print(f"Dataset Size: {len(base_dataset):,}")
        print(f"Model Architecture:\n{model_config}")
        print(f"Log Directory: {LOG_DIR}")

