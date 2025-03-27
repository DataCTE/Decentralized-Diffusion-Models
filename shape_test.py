import os
import tempfile
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
from config import get_config
from data.dataset import DDMDataset
from models.mmdit import ExpertMMDiT
from models.router import RouterModel
from trainers.sampling import ddm_sample
from trainers.router import RouterTrainer
from trainers.expert import ExpertTrainer
from utils.distributed import get_rank, is_main_process
from utils.logging import setup_logger
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F
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
        'num_clusters': 4,
        'expert_capacity_factor': 1.2,
        'router_temperature': 2.0,
        'router_min_temp': 0.5,
        'router_temperature_decay': 0.99995,
        'sampling_steps': 10
    }
    config = SimpleNamespace(**config_dict)

    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup dataset paths
        config.feature_cache_path = os.path.join(tmpdir, "cache")
        os.makedirs(config.feature_cache_path, exist_ok=True)
        
        # Create dummy features and clusters
        num_dummy_images = 200
        torch.save(torch.randn(num_dummy_images, 1024), 
                 os.path.join(config.feature_cache_path, "train_features.pt"))
        torch.save(torch.randint(0,4,(num_dummy_images,)), 
                 os.path.join(config.feature_cache_path, "final_clusters.pt"))

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
            'cluster_labels': torch.randint(0,4,(2,), device=device)
        }
        
        # Paper's recommended router warmup
        print("Testing Router Training:")
        router_losses = []
        for step in range(20):
            loss = router_trainer.train_step(batch)
            router_losses.append(loss)
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
            return torch.randn(2, config.num_experts, device=device)

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
