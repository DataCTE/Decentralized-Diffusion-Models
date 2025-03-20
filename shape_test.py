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
from models.dit import ExpertDiT
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup paths
        config.dataset_path = tmpdir
        train_dir = os.path.join(tmpdir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create dummy dataset (4 images)
        dummy_images = [np.random.rand(256, 256, 3) * 255 for _ in range(4)]
        dummy_features = torch.randn(4, 1024)  # Fake DINOv2 features
        
        # Save dummy data in paper's format
        for i in range(4):
            img = Image.fromarray(dummy_images[i].astype('uint8'))
            img.save(os.path.join(train_dir, f"image_{i}.png"))
            
        # Save features and clusters
        torch.save(dummy_features, os.path.join(tmpdir, "train_features.pt"))
        torch.save(torch.randint(0,4,(4,)), os.path.join(tmpdir, "train_clusters.pt"))
        
        # Initialize dataset
        dataset = DDMDataset(config, split='train')
        
        # Test clustering matches paper specs
        assert hasattr(dataset, 'clusterer'), "Clustering not initialized"
        assert dataset.clusterer.num_coarse == config.num_experts, "Cluster count mismatch"
        assert dataset.clusterer.num_fine == 1024, "Fine clusters not 1024"
        
        # Initialize models with paper's architecture
        router = RouterModel(config).to(device)
        experts = {i: ExpertDiT(config).to(device) for i in range(config.num_experts)}
        
        # Test router forward pass
        dummy_latent = torch.randn(1, config.latent_channels, 16, 16, device=device)
        dummy_t = torch.randint(0, 1000, (1,), device=device)
        router_logits = router(dummy_latent, dummy_t)
        assert router_logits.shape == (1, config.num_experts), "Router output shape mismatch"
        
        # Test expert forward pass
        expert_out = experts[0](dummy_latent, dummy_t)
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
