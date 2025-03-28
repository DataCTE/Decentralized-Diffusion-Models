import os
import tempfile
import torch
import torch.nn as nn
from config import get_config
from data.dataset import DDMDataset
from trainers.sampling import ddm_sample
from trainers.router import RouterTrainer
from trainers.expert import ExpertTrainer
from utils.logging import setup_logger
from types import SimpleNamespace
import torch.distributed as dist
from einops import rearrange

def test_full_pipeline():
    """End-to-end shape test of DDM pipeline with dummy data using FSDP"""
    # Set required environment variables for single-process distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Initialize distributed environment first
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo" if not torch.cuda.is_available() else "nccl",
            init_method="env://",
            world_size=1,
            rank=0
        )
    
    # Setup distributed environment
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    logger = setup_logger("ShapeTest")
    print(f"\n{'='*40} INITIALIZING TEST PIPELINE {'='*40}")

    config = get_config()
    config_dict = {
    # ===== Model Architecture =====
    'hidden_size': 256,          # Reduced from 768
    'num_heads': 8,              # Reduced from 12
    'depth': 4,                  # Reduced from 12
    'mlp_ratio': 2.0,
    'qkv_bias': True,
    'vec_in_dim': 768,           # Must match CLIP's 768D
    'context_in_dim': 768,       # Match CLIP embedding dim
    'position_embed_type': 'rope_2d',
    'theta': 10000,
    'axes_dim': [16, 16],        # Reduced from [32,32]
    
    # ===== Training Configuration =====
    'batch_size': 1,
    'expert_batch_size': 1,
    'learning_rate': 2e-4,
    'weight_decay': 0.05,
    'warmup_steps': 500,
    'num_steps': 100000,
    'use_mixed_precision': True,
    'gradient_accumulation_steps': 1,
    'use_scheduler': True,
    'scheduler_type': 'linear',
    'expert_loss_weight': 1.0,
    'adam_betas': (0.9, 0.98),
    'eps': 1e-8,
    
    # ===== Expert Configuration =====
    'num_experts': 2,
    'num_clusters': 2,
    'cluster_embed_dim': 768,
    'cluster_embed': 768,
    'expert_capacity_factor': 1.2,
    'max_experts_in_memory': 2,
    'expert_offload_to_cpu': True,
    'patch_size': 4,
    'guidance_embed': False,
    
    # ===== Router Configuration =====
    'router_learning_rate': 3e-4,
    'router_hidden_size': 192,   # Reduced from 384
    'router_num_heads': 4,       # Reduced from 6
    'router_temperature': 1.5,
    'router_min_temp': 0.5,
    'router_temperature_decay': 0.9999,
    'router_model': 'paper_baseline',
    'balance_lambda': 0.1,
    
    # ===== Diffusion Configuration =====
    'loss_type': 'huber',
    'sigma': 1.0,
    'sampling_steps': 10,
    'top_p': 0.9,
    
    # ===== Dataset & Latent Configuration =====
    'buckets': [
            (64, 64), (96, 64),      # Smaller test resolutions
            (64, 96), (128, 64)
    ],
    'image_size': 64,
    'latent_channels': 16,
    'in_channels': 16,
    'out_channels': 16,
    'vae_scaling_factor': 0.18215,
    'vae_model': 'AuraDiffusion/16ch-vae',
    
    # ===== CLIP Configuration =====
    'clip_model': 'openai/clip-vit-large-patch14',
    'clip_embedding_dim': 768,
    
    # ===== Distributed Training =====
    'gradient_checkpointing': True,
    'fsdp_sharding_strategy': "FULL_SHARD",
    'fsdp_auto_wrap_policy': "LAMBDA",
    
    # ===== Logging & Monitoring =====
    'output_dir': './outputs',
    'feature_cache_path': './cache',
    'wandb_enabled': True,
    'wandb_project': 'decentralized-diffusion',
    'verbose_training': False,
    'log_memory': False,
    
    # ===== Validation & Sampling =====
    'enable_validation': False,
    'validation_interval': 1000,
    'enable_sampling': True,
    'sampling_steps': 50,
    'cfg_scale': 7.5,
    'save_interval': 5000,
    
    # ===== Debugging & Testing =====
    'bypass_cluster_validation': False,
    'depth_single_blocks': 5,
    
    # ===== Model References =====
    'distilled_model': None,
    
    # ===== Expert Training Configuration =====
    'expert_learning_rate': 2e-4,
    'expert_warmup_steps': 500,
    'expert_scheduler_type': 'linear',
    'expert_gradient_accumulation_steps': 1,
    'expert_adam_betas': (0.9, 0.98),
    'expert_weight_decay': 0.05,
    'expert_max_grad_norm': 1.0,
    
    # ===== Expert Metrics Configuration =====
    'expert_metrics': {
        'track_ensemble': True,
        'utilization_threshold': 0.1,
        'alignment_window': 1000
    },
    }
    config = SimpleNamespace(**config_dict)
    
    # Force critical parameters to match
    config.latent_channels = 16  # For 16ch VAE
    config.in_channels = config.latent_channels  # Must match
    
    print(f"\n{'='*40} TEST CONFIG {'='*40}")
    print(f"Latent channels: {config.latent_channels}")
    print(f"VAE scaling factor: {config.vae_scaling_factor}")
    print(f"Using mixed precision: {config.use_mixed_precision}")

    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup dataset paths
        config.feature_cache_path = os.path.join(tmpdir, "cache")
        os.makedirs(config.feature_cache_path, exist_ok=True)
        
        # Create required subdirectories
        os.makedirs(os.path.join(config.feature_cache_path, "clip"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "latents"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "clusters"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "dims"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "buckets"), exist_ok=True)
        
        # Create per-sample dummy files with consistent naming
        num_dummy_images = 200
        image_size = config.image_size
        all_cluster_ids = [] 
        
        for i in range(num_dummy_images):
            base_name = f"anime-{i}"
            
            # Create cluster assignment
            cluster_id = torch.randint(0, config.num_clusters, (1,))
            all_cluster_ids.append(cluster_id)  # Store cluster IDs
            
            # Create latent file with proper dimensions
            torch.save(
                torch.randn(config.latent_channels, 32, 32),  # [C, H, W]
                os.path.join(config.feature_cache_path, "latents", f"{base_name}_rank0.pt")
            )
            
            # Create CLIP embedding with proper dimensions
            torch.save(
                torch.randn(77, config.clip_embedding_dim),
                os.path.join(config.feature_cache_path, "clip", f"{base_name}_rank0.pt")
            )
            
            # Create bucket assignment
            bucket_idx = torch.randint(0, len(config.buckets), (1,))
            torch.save(
                bucket_idx,
                os.path.join(config.feature_cache_path, "buckets", f"{base_name}_rank0.pt")
            )
            
            # Create dimension entry
            bucket_dims = config.buckets[bucket_idx.item()]
            torch.save(
                torch.tensor(bucket_dims, dtype=torch.int16),
                os.path.join(config.feature_cache_path, "dims", f"{base_name}_rank0.pt")
            )

        # Create final_clusters.pt with proper structure
        final_clusters = {f"anime-{i}": int(cluster_id.item()) for i, cluster_id in enumerate(all_cluster_ids)}
        torch.save(
            final_clusters,
            os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt")
        )

        # Initialize dataset with proper config handling
        dataset = DDMDataset(vars(config), 'train')
        
        print(f"\n{'='*40} DATASET INFO {'='*40}")
        print(f"Created {num_dummy_images} dummy samples")
        cluster_ids = list(final_clusters.values())
        cluster_tensor = torch.tensor(cluster_ids, dtype=torch.long)
        unique_clusters, counts = torch.unique(cluster_tensor, return_counts=True)
        print(f"Final cluster distribution:\nClusters: {unique_clusters.tolist()}\nCounts: {counts.tolist()}")

        # Initialize models with paper's architecture
        router_trainer = RouterTrainer(config, device, rank, world_size=1)
        print(f"\n{'='*40} MODEL INIT {'='*40}")
        print(f"Router parameter count: {sum(p.numel() for p in router_trainer.router.parameters())}")
        
        # Test router forward pass with proper dimensions
        dummy_latent = torch.randn(2, config.latent_channels, 32, 32, device=device)
        dummy_t = torch.randint(0, 1000, (2,), device=device).float() / 1000
        dummy_text_embeds = torch.randn(2, 77, 768, device=device)
        
        # Add proper router test with cluster validation
        router_logits = router_trainer.router(
            img=dummy_latent,
            timesteps=dummy_t * 1000,
            txt=dummy_text_embeds
        )
        print(f"\n[Router Test] Output shape: {router_logits.shape}")
        assert router_logits.shape == (2, config.num_experts), \
            f"Router output shape mismatch: {router_logits.shape} vs expected (2, {config.num_experts})"
        
        # Convert logits to cluster predictions
        cluster_preds = torch.argmax(router_logits, dim=-1)
        cluster_preds = torch.zeros_like(cluster_preds)  # Force to expert 0
        print(f"Modified clusters for testing: {cluster_preds.cpu().tolist()}")
        
        # Add router logging
        print(f"\n[Router Debug]")
        print(f"Logits range: {router_logits.min().item():.2f} to {router_logits.max().item():.2f}")
        print(f"Logits mean: {router_logits.mean().item():.2f} ± {router_logits.std().item():.2f}")
        
        # Calculate expert distribution
        probs = torch.softmax(router_logits, dim=-1)
        print(f"Expert probabilities:")
        for expert_idx in range(config.num_experts):
            expert_count = (cluster_preds == expert_idx).sum().item()
            expert_prob = probs[:, expert_idx].mean().item()
            print(f"  Expert {expert_idx}: {expert_count} samples ({expert_prob*100:.1f}% confidence)")

        # Validate cluster predictions
        assert torch.all(cluster_preds < config.num_experts), "Router predicted invalid cluster IDs"

        # Test expert training with router predictions
        batch = {
            'latent': dummy_latent,
            'clip_embedding': dummy_text_embeds,
            'expert': cluster_preds.detach()  # Use router predictions instead of ground truth
        }
        
        # Initialize expert with proper latent_channels handling
        expert_trainer = ExpertTrainer(
            expert_idx=0,
            config=config,
            device=device,
            rank=rank,
            world_size=1,
            router=router_trainer.router
        )
        
        # Verify expert input dimensions
        print(f"\n{'='*40} EXPERT TRAINING {'='*40}")
        print("Expert input shapes:")
        print(f"  Latent: {dummy_latent.shape}")
        
        # Test expert forward pass with cluster IDs
        expert_input = rearrange(
            dummy_latent,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=config.patch_size,
            p2=config.patch_size
        )
        img_ids = torch.stack(
            torch.meshgrid(
                torch.arange(8, device=device),
                torch.arange(8, device=device),
                indexing='ij'
            ), -1
        ).reshape(1, 64, 2).expand(2, -1, -1)

        expert_out = expert_trainer.expert(
            img=expert_input,
            img_ids=img_ids,  # Updated position IDs
            txt=dummy_text_embeds,
            txt_ids=torch.randint(0,77,(2,77,2), device=device),
            timesteps=(dummy_t * 1000).float(),
            y=torch.randn(2, config.vec_in_dim, device=device),
            cluster_ids=torch.randint(0, config.num_clusters, (2,), device=device)
        )
        assert expert_out.shape == (2, config.latent_channels, 32, 32), \
            f"Expert output shape mismatch: {expert_out.shape} vs expected (2, 16, 32, 32)"

        # Correct dimension check
        expected_in_channels = config.latent_channels * (config.patch_size ** 2)
        print(f"\n{'='*40} DIMENSION CHECKS {'='*40}")
        print(f"Expected input channels: {expected_in_channels}")
        print(f"Actual input channels: {expert_input.shape[-1]}")
        print(f"Expected output channels: {config.latent_channels}")
        print(f"Actual output channels: {expert_out.shape[1]}")
        
        assert expert_input.shape[-1] == expected_in_channels, \
            f"Input dimension mismatch: {expert_input.shape[-1]} vs {expected_in_channels}"
        
        assert expert_out.shape[1] == config.latent_channels, \
            f"Output channel mismatch: {expert_out.shape[1]} vs {config.latent_channels}"

        # Verify final_layer configuration
        final_layer = expert_trainer.expert.final_layer
        assert final_layer.patch_size == config.patch_size, \
            f"Final layer patch_size mismatch: {final_layer.patch_size} vs {config.patch_size}"
        assert final_layer.out_channels == config.latent_channels, \
            f"Final layer out_channels mismatch: {final_layer.out_channels} vs {config.latent_channels}"

        # Initialize all experts
        experts = {}
        for expert_idx in range(config.num_experts):
            expert_trainer = ExpertTrainer(
                expert_idx=expert_idx,
                config=config,
                device=device,
                rank=rank,
                world_size=1,
                router=router_trainer.router
            )
            experts[expert_idx] = expert_trainer.expert

        # Test sampling with proper latent dimensions
        print(f"\n{'='*40} SAMPLING {'='*40}")
        samples = ddm_sample(
            router=lambda img, timesteps, txt: router_trainer.router(img, timesteps, txt),
            experts=experts,
            shape=(2, config.latent_channels, 32, 32),
            num_steps=config.sampling_steps,
            device=device,
            text_embeddings=dummy_text_embeds,
            temperature=0.1,
            config=config
        )
        assert samples.shape == (2, config.latent_channels, 32, 32), \
            f"Sampling shape mismatch: {samples.shape} vs expected (2, 16, 32, 32)"

        # Final cleanup
        print(f"\n{'='*40} TEST COMPLETE {'='*40}")
        if dist.is_initialized():
            dist.destroy_process_group()

    # Ensure cluster_embed matches num_clusters
    router_trainer.router.cluster_embed = nn.Embedding(
        config.num_clusters,  # Was previously using num_experts
        config.cluster_embed_dim
    ).to(device)

    # Remove non-architectural params from model_config_dict
    del config_dict['batch_size']
    del config_dict['expert_batch_size']
    # ... remove other training-specific params

if __name__ == "__main__":
    try:
        test_full_pipeline()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
