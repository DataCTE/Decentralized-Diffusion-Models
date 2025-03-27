import os
import tempfile
import torch
import torch.nn.functional as F
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

def test_full_pipeline():
    """End-to-end shape test of DDM pipeline with dummy data using FSDP"""
    # Setup distributed environment
    rank = get_rank()
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
        
        # Create dummy features and clusters
        num_dummy_images = 200
        image_size = config.image_size
        
        # Create dimension files
        torch.save(
            torch.tensor([image_size]*num_dummy_images, dtype=torch.int16),
            os.path.join(config.feature_cache_path, "dims", "train_features.pt")
        )
        
        # Create bucket assignments
        num_buckets = len(config.buckets)
        torch.save(
            torch.randint(0, num_buckets, (num_dummy_images,), dtype=torch.int16),
            os.path.join(config.feature_cache_path, "buckets", "train_features.pt")
        )

        # Create dummy features and clusters
        torch.save(torch.randn(num_dummy_images, 1024), 
                 os.path.join(config.feature_cache_path, "train_features.pt"))
        torch.save(torch.randint(0,4,(num_dummy_images,)), 
                 os.path.join(config.feature_cache_path, "clusters", "final_clusters.pt"))

        # Create dummy text embeddings
        torch.save(
            torch.randn(num_dummy_images, 77, config.clip_embedding_dim),
            os.path.join(config.feature_cache_path, "clip", "train_features.pt")
        )
        torch.save(
            torch.randn(num_dummy_images, 128, config.t5_embedding_dim),
            os.path.join(config.feature_cache_path, "t5", "train_features.pt")
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

if __name__ == "__main__":
    test_full_pipeline()
