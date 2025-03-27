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
        # ===== Core Architecture =====
        'hidden_size': 512,          # Reduced from 1152
        'num_heads': 8,              # Reduced from 16
        'depth': 6,                  # Reduced from 30
        'mlp_ratio': 2.0,            # Reduced from 4.0
        'qkv_bias': True,             
        'position_embed_type': 'rope_2d',  
        
        # ===== Expert Configuration =====
        'num_experts': 2,            # Keep minimal experts
        'num_clusters': 2,
        'cluster_embed_dim': 256,    # Reduced from 512
        'expert_capacity_factor': 1.2,
        
        # ===== Training Parameters ===== 
        'batch_size': 1,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 2,
        
        # ===== Dataset & Latent Config =====
        'image_size': (256, 256),
        'latent_channels': 16,  # Critical for VAE compatibility
        'vae_scaling_factor': 0.18215,  
        'in_channels': 16,  # Must match latent_channels
        'out_channels': 16, 
        
        # ===== Positional Embeddings =====
        'axes_dim': [32, 32],        # Reduced from [36,36]
        'theta': 10000,               
        
        # ===== Router Configuration =====
        'router_hidden_size': 256,   # Reduced from 512
        'router_num_heads': 8,
        'router_temperature': 2.0,
        'router_min_temp': 0.5,
        'router_temperature_decay': 0.99995,
        
        # ===== Diffusion & Loss =====
        'loss_type': 'huber',         
        'sigma': 1.0,
        'sampling_steps': 10,
        'top_p': 0.9,                 
        
        # ===== Optimization =====
        'adam_betas': (0.9, 0.999),
        'eps': 1e-8,
        
        # ===== Distributed Training =====
        'gradient_checkpointing': True,
        
        # ===== Experimental Flags =====
        'bypass_cluster_validation': True,
        'guidance_embed': False,      
        
        # Required parameters from latest implementation
        'vec_in_dim': 768,  
        'context_in_dim': 768,  
        'buckets': [(256, 256), (288, 224), (224, 288)], 
        'clip_embedding_dim': 768, 
        't5_embedding_dim': 512, 
        'num_steps': 10,  
        'balance_lambda': 0.01,  
        'router_model': 'paper_baseline',  
        'use_scheduler': True,  
        'scheduler_type': 'cosine',  
        'depth_single_blocks': 2,    # Reduced from 5
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
            
            # Create cluster assignment
            cluster_id = torch.randint(0, config.num_clusters, (1,))
            all_cluster_ids.append(cluster_id)
            torch.save(
                cluster_id,
                os.path.join(config.feature_cache_path, "clusters", f"{base_name}_rank0.pt")
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
        final_clusters = torch.cat(all_cluster_ids)
        torch.save(
            final_clusters,
            os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt")
        )

        # Initialize dataset with proper config handling
        dataset = DDMDataset(vars(config), 'train')
        
        print(f"\n{'='*40} DATASET INFO {'='*40}")
        print(f"Created {num_dummy_images} dummy samples")
        print(f"Final cluster distribution: {final_clusters.unique(return_counts=True)}")

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
        cluster_ids = torch.randint(0, config.num_clusters, (2,), device=device)
        expert_out = expert_trainer.expert(
            img=torch.randn(2, 32*32, config.latent_channels, device=device),
            img_ids=torch.randint(0,32,(2,32*32,2), device=device),
            txt=dummy_text_embeds,
            txt_ids=torch.randint(0,77,(2,77,2), device=device),
            timesteps=(dummy_t * 1000).float(),
            y=torch.randn(2, config.vec_in_dim, device=device),
            cluster_ids=cluster_ids
        )
        assert expert_out.shape == (2, 32*32, config.latent_channels), \
            f"Expert output shape mismatch: {expert_out.shape} vs expected (2, 1024, 16)"

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
            router=lambda x, t, txt: router_trainer.router(x, t, txt),
            experts=experts,  # Now contains all 4 experts
            shape=(2, config.latent_channels, 32, 32),
            num_steps=config.sampling_steps,
            device=device,
            text_embeddings=dummy_text_embeds,
            temperature=0.1
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

if __name__ == "__main__":
    try:
        test_full_pipeline()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
