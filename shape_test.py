import os
import tempfile
import torch
import torch.nn.functional as F
import torch.nn as nn
from config import get_config
from data.dataset import DDMDataset
from models.mmdit import ExpertMMDiT
from models.router import RouterModel
from trainers.sampling import ddm_sample
from trainers.router import RouterTrainer
from trainers.expert import ExpertTrainer
from utils.distributed import get_rank, is_main_process
from utils.logging import setup_logger
from types import SimpleNamespace
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

def test_full_pipeline():
    """End-to-end shape test of DDM pipeline with dummy data using FSDP"""
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

    config = get_config()
    config_dict = {
        'num_experts': 4,
        'batch_size': 2,
        'image_size': (256, 256),
        'latent_channels': 16,
        'bypass_cluster_validation': True,
        'buckets': [(256, 256), (288, 224), (224, 288)],
        'clip_embedding_dim': 768,
        't5_embedding_dim': 512,
        'num_clusters': 4,
        'expert_capacity_factor': 1.2,
        'router_temperature': 2.0,
        'router_min_temp': 0.5,
        'router_temperature_decay': 0.99995,
        'sampling_steps': 10,
        'router_hidden_size': 512,
        'router_num_heads': 8,
        'top_p': 0.9,  # For nucleus sampling
        'router_model': 'paper_baseline',
    }
    config = SimpleNamespace(**config_dict)

    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup dataset paths
        config.feature_cache_path = os.path.join(tmpdir, "cache")
        os.makedirs(config.feature_cache_path, exist_ok=True)
        
        # Create required subdirectories
        os.makedirs(os.path.join(config.feature_cache_path, "clip"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "t5"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "latents"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "clusters"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "dims"), exist_ok=True)
        os.makedirs(os.path.join(config.feature_cache_path, "buckets"), exist_ok=True)
        
        # Create per-sample dummy files with consistent naming
        num_dummy_images = 200
        image_size = config.image_size
        all_cluster_ids = []  # Add this line to collect cluster IDs
        
        for i in range(num_dummy_images):
            base_name = f"anime-{i}"
            
            # Create latent file
            torch.save(
                torch.randn(config.latent_channels, 32, 32),  # Match latent_channels
                os.path.join(config.feature_cache_path, "latents", f"{base_name}_rank0.pt")
            )
            
            # Create CLIP embedding
            torch.save(
                torch.randn(77, config.clip_embedding_dim),
                os.path.join(config.feature_cache_path, "clip", f"{base_name}_rank0.pt")
            )
            
            # Create T5 embedding 
            torch.save(
                torch.randn(128, config.t5_embedding_dim),
                os.path.join(config.feature_cache_path, "t5", f"{base_name}_rank0.pt")
            )
            
            # Create cluster assignment
            cluster_id = torch.randint(0, config.num_clusters, (1,))
            all_cluster_ids.append(cluster_id)  # Collect cluster IDs
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

        # Add this block after the loop to create final_clusters.pt
        final_clusters = torch.cat(all_cluster_ids)
        torch.save(
            final_clusters,
            os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt")
        )

        # Initialize dataset with proper config handling
        dataset = DDMDataset(vars(config), 'train')
        
        # Initialize models with paper's architecture
        router_trainer = RouterTrainer(config, device, rank, world_size=1)
        
        # Test router forward pass
        dummy_latent = torch.randn(2, config.latent_channels, 32, 32, device=device)
        dummy_t = torch.randint(0, 1000, (2,), device=device)
        dummy_text_embeds = torch.randn(2, 77, 768, device=device)
        
        # Test router training step
        batch = {
            'latent': dummy_latent,
            'clip_embedding': dummy_text_embeds,
            'cluster_labels': torch.load(
                os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt")
            )[:2].to(device)
        }
        
        # Paper's recommended router warmup
        print("Testing Router Training:")
        router_losses = []
        for step in range(20):
            loss = router_trainer.train_step(batch)
            router_losses.append(loss)
            
            with torch.no_grad():
                logits = router_trainer.router(
                    batch['latent'],
                    torch.randint(0, 1000, (2,), device=device),
                    batch['clip_embedding']
                )
                assert logits.shape == (2, config.num_experts), \
                    f"Bad router shape: {logits.shape}"
                
                probs = torch.softmax(logits, dim=-1)
                assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device)), \
                    "Router outputs not valid probabilities"
            
            print(f"Step {step+1}/20 - Loss: {loss:.4f}")

        # Initialize expert trainer with paper's config
        expert_trainer = ExpertTrainer(
            expert_idx=0,
            config=config,
            device=device,
            rank=rank,
            world_size=1,
            router=router_trainer.router
        )
        
        # Test expert forward pass
        cluster_ids = torch.randint(0,4,(2,), device=device)
        expert_out = expert_trainer.expert(
            img=torch.randn(2, 32*32, config.latent_channels, device=device),
            img_ids=torch.randint(0,32,(2,32*32,2), device=device),
            txt=dummy_text_embeds,
            txt_ids=torch.randint(0,77,(2,77,2), device=device),
            timesteps=dummy_t*1000,
            y=torch.randn(2, config.vec_in_dim, device=device),
            cluster_ids=cluster_ids
        )
        assert expert_out.shape == (2, 32*32, config.latent_channels), "Expert output shape mismatch"

        # Test expert training
        print("Testing Expert Training:")
        expert_losses = []
        for step in range(20):
            loss = expert_trainer.train_step({
                'latent': dummy_latent,
                'clip_embedding': dummy_text_embeds,
                'cluster_pred': cluster_ids
            })
            expert_losses.append(loss)
            print(f"Step {step+1}/20 - Loss: {loss:.4f}")

        # Test sampling with paper's recommended settings
        def dummy_router(x_t, timestep, text_embeddings):
            with torch.no_grad():
                cluster_preds = torch.load(
                    os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt")
                )[:2].to(device)
                
                return F.one_hot(cluster_preds, num_classes=config.num_experts).float()

        experts = {0: expert_trainer.expert}
        
        samples = ddm_sample(
            router=dummy_router,
            experts=experts,
            shape=(2, config.latent_channels, 32, 32),
            num_steps=config.sampling_steps,
            device=device,
            text_embeddings=dummy_text_embeds,
            inference_strategy="top_k",
            top_k=1
        )
        assert samples.shape == (2, config.latent_channels, 32, 32), "Sampling shape mismatch"

        # Test router shapes
        def test_router_shapes():
            """Validate router input/output shapes and training behavior"""
            # Test initialization
            assert hasattr(router_trainer, 'router'), "Router not initialized"
            assert isinstance(router_trainer.router, (nn.Module, FSDP)), "Invalid router type"
            
            # Test training forward pass
            with torch.no_grad():
                # Get initial predictions
                logits = router_trainer.router(
                    batch['latent'],
                    torch.randint(0, 1000, (2,), device=device),
                    batch['clip_embedding']
                )
                # Validate output shape
                assert logits.shape == (2, config.num_experts), \
                    f"Bad router logits shape: {logits.shape}"
                    
                # Validate probability calculations
                probs = torch.softmax(logits, dim=-1)
                assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=device)), \
                    "Router outputs not valid probabilities"

            # Test inference modes
            for strategy in ["top_k", "nucleus", "random"]:
                expert_weights = router_trainer.get_expert_weights(
                    batch['latent'],
                    torch.randint(0, 1000, (2,), device=device),
                    batch['clip_embedding'],
                    strategy=strategy,
                    k=1 if strategy == "top_k" else None
                )
                
                # Validate expert weights
                assert expert_weights.shape == (2, config.num_experts), \
                    f"Bad expert weights shape for {strategy}"
                assert torch.allclose(expert_weights.sum(dim=1), torch.ones(2, device=device)), \
                    f"Expert weights don't sum to 1 for {strategy}"

            # Test temperature scaling
            original_logits = router_trainer.router(
                batch['latent'],
                torch.randint(0, 1000, (2,), device=device),
                batch['clip_embedding']
            )
            
            # Change temperature and verify effect
            router_trainer.router.temperature = 0.5
            scaled_logits = router_trainer.router(
                batch['latent'],
                torch.randint(0, 1000, (2,), device=device),
                batch['clip_embedding']
            )
            
            # Validate temperature scaling
            assert not torch.allclose(original_logits, scaled_logits), \
                "Temperature scaling not working"
            assert torch.allclose(original_logits / 0.5, scaled_logits), \
                "Incorrect temperature scaling implementation"

        test_router_shapes()

if __name__ == "__main__":
    test_full_pipeline()
    # Cleanup distributed after test
    if dist.is_initialized():
        dist.destroy_process_group()
