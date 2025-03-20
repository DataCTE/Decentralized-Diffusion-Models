import os
import tempfile
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
from config import get_config
from data.dataset import DDMDataset
from data.clustering import DDMClustering
from models.mmdit import ExpertMMDiT
from models.router import RouterModel
from trainers.sampling import ddm_sample
from trainers.router import RouterTrainer
from trainers.expert import ExpertTrainer
from trainers.diffusion import DecentralizedFlowMatcher

def test_full_pipeline():
    """End-to-end shape test of DDM pipeline with dummy data"""
    # Setup
    config = get_config("config.py")
    config.num_experts = 4
    config.batch_size = 2
    config.image_size = (64, 64)
    config.latent_channels = 16
    config.bypass_cluster_validation = True # Enable bypass for shape test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup paths
        config.dataset_path = tmpdir
        train_dir = os.path.join(tmpdir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create dummy dataset (200 images)
        num_dummy_images = 2000
        dummy_images = [np.random.rand(256, 256, 3) * 255 for _ in range(num_dummy_images)]
        dummy_features = torch.randn(num_dummy_images, 1024)  # Fake DINOv2 features
        dummy_dims = torch.tensor([[256, 256] for _ in range(num_dummy_images)], dtype=torch.int64)
        dummy_dims = dummy_dims.reshape(num_dummy_images, 2)  # Explicitly reshape to [num_dummy_images, 2]
        
        # Save dummy data in paper's format
        for i in range(num_dummy_images):
            img = Image.fromarray(dummy_images[i].astype('uint8'))
            img.save(os.path.join(train_dir, f"image_{i}.png"))
            caption_path = os.path.join(train_dir, f"image_{i}.txt")
            with open(caption_path, 'w') as f:
                f.write("dummy caption")
            
        # Save features and clusters
        torch.save(dummy_features, os.path.join(tmpdir, "train_features.pt"))
        torch.save(dummy_dims, os.path.join(tmpdir, "dim_cache.pt"))
        torch.save(torch.randint(0,4,(num_dummy_images,)), os.path.join(tmpdir, "train_clusters.pt"))
        
        # Set num_fine_clusters in config before dataset initialization
        config.num_fine_clusters = 2
        print(f"config.num_fine_clusters in shape_test: {config.num_fine_clusters}")
        # Initialize dataset
        dataset = DDMDataset(config, split='train')
        
        # Test clustering matches paper specs
        assert hasattr(dataset, 'clusterer'), "Clustering not initialized"
        assert dataset.clusterer.num_coarse == config.num_experts, "Cluster count mismatch"
        assert dataset.clusterer.num_fine == 2, "Fine clusters not 2"
        
        # Test router temperature decay
        def test_router_temperature_decay():
            router_temp_test = RouterModel(config).to(device)
            initial_temp = router_temp_test.get_temperature()
            assert initial_temp == 2.0, "Initial temperature should be 2.0"

            # Simulate some training steps to decay temperature
            for _ in range(1000):
                router_temp_test.update_temperature()

            decayed_temp = router_temp_test.get_temperature()
            assert decayed_temp < initial_temp, "Temperature should decay"
            assert decayed_temp >= 0.5, "Temperature should not go below 0.5"
            print("Router temperature decay test passed!")

        test_router_temperature_decay()
        
        # Initialize models with paper's architecture
        router = RouterModel(config).to(device)
        experts = {i: ExpertMMDiT(config).to(device) for i in range(config.num_experts)}
        
        # Test router forward pass
        dummy_latent = torch.randn(1, config.latent_channels, 16, 16, device=device)
        dummy_t = torch.randint(0, 1000, (1,), device=device)
        dummy_text_embeds = torch.randn(1, config.clip_embedding_dim, device=device)
        router_logits = router(dummy_latent, dummy_t, dummy_text_embeds)
        assert router_logits.shape == (1, config.num_experts), "Router output shape mismatch"
        
        # Test expert forward pass
        dummy_text_embeds = torch.randn(1, 768, device=device)
        expert_out = experts[0](dummy_latent, dummy_t, dummy_text_embeds)
        assert expert_out.shape == dummy_latent.shape, "Expert output shape mismatch"
        
        # Test training steps
        # Router training
        router_trainer = RouterTrainer(config, device, rank=0, world_size=1)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        router_loss = router_trainer.train_step(batch)
        assert isinstance(router_loss.item(), float), "Router training failed"
        
        # Expert training
        expert_trainer = ExpertTrainer(0, config, device, 0, 1)
        expert_loss = expert_trainer.train_step(batch)
        assert isinstance(expert_loss.item(), float), "Expert training failed"
        
        # Test sampling pipeline
        shape = (2, config.latent_channels, 16, 16)
        samples = ddm_sample(
            router=router,
            experts=experts,
            shape=shape,
            steps=4,
            top_k=1,
            device=device,
            config=config
        )
        assert samples.shape == shape, "Sampling output shape mismatch"
        print("All shape tests passed!")

if __name__ == "__main__":
    test_full_pipeline()
