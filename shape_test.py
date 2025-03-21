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
from utils.distributed import setup_distributed, get_rank
from utils.fsdp import create_fsdp_model
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F

def test_full_pipeline():
    """End-to-end shape test of DDM pipeline with dummy data using FSDP"""
    # Setup distributed environment for FSDP
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    config = get_config("config.py")
    config.num_experts = 4
    config.batch_size = 2
    config.image_size = (256, 256)
    config.latent_channels = 16
    config.bypass_cluster_validation = True # Enable bypass for shape test

    # Modify config to include multiple buckets
    config.buckets = [
        (256, 256),   # Original size
        (288, 224),   # Landscape variant
        (224, 288),   # Portrait variant
    ]
    config.image_size = (256, 256)  # Base size
    config.latent_channels = 16
    config.bypass_cluster_validation = True

    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup paths
        config.dataset_path = tmpdir
        train_dir = os.path.join(tmpdir, "train")
        os.makedirs(train_dir, exist_ok=True)
        
        # Create dummy dataset (200 images)
        num_dummy_images = 2000
        dummy_images = []
        bucket_sizes = [(256, 256), (288, 224), (224, 288)]
        for i in range(num_dummy_images):
            # Assign images to different buckets
            w, h = bucket_sizes[i % 3]
            dummy_images.append(np.random.rand(h, w, 3) * 255)
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
        
        # Save clustered features in paper's format
        cluster_dir = os.path.join(tmpdir, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Paper's clustering: 2-stage process with fine then coarse
        fine_clusters = torch.randint(0, 2, (num_dummy_images,))  # Stage 1: 2 fine clusters
        coarse_clusters = torch.randint(0, 4, (num_dummy_images,))  # Stage 2: 4 experts
        
        # Save in paper's expected format - 1 file per coarse cluster
        for cluster_idx in range(config.num_experts):
            mask = coarse_clusters == cluster_idx
            torch.save({
                'fine': fine_clusters[mask],
                'coarse': coarse_clusters[mask]
            }, os.path.join(cluster_dir, f"cluster_{cluster_idx}.pt"))
        
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
        
        # Initialize models with paper's architecture, now FSDP wrapped
        base_router = RouterModel(config)
        router = create_fsdp_model(base_router, config, rank=rank).to(device)

        experts = {}
        for i in range(config.num_experts):
            base_expert = ExpertMMDiT(config)
            experts[i] = create_fsdp_model(base_expert, config, rank=rank).to(device)
        
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

        print("Testing Router Training for 20 steps:")
        router_losses = []
        for step in range(20):  # Run router training for 20 steps
            router_loss = router_trainer.train_step(batch)
            router_losses.append(router_loss)
            print(f"  Step {step+1}/20 - Router Loss: {router_loss:.4f}")
        assert isinstance(router_loss, float), "Router training failed"

        # Expert training
        expert_trainer = ExpertTrainer(0, config, device, 0, 1)

        # Add explicit test for flow matching with different sizes
        def test_bucket_aware_flow_matching():
            flow_matcher = DecentralizedFlowMatcher(loss_type='huber')
            
            # Create mismatched prediction/target sizes
            pred = torch.randn(2, 16, 32, 32)  # Simulate 256x256 bucket
            target = torch.randn(2, 16, 28, 28)  # Simulate 224x288 bucket
            
            try:
                loss = flow_matcher.compute_flow_matching_loss(pred, target)
                assert isinstance(loss.item(), float), "Should return scalar loss"
                print("Bucket-aware flow matching test passed!")
            except Exception as e:
                assert False, f"Bucket-aware flow matching failed: {str(e)}"

        test_bucket_aware_flow_matching()
        
        # Modify expert training test to verify bucket handling
        print("Testing Expert Training for 20 steps with mixed buckets:")
        expert_losses = []
        for step in range(20):
            # Get batch with mixed buckets (simulated via resizing)
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            
            # Manually modify batch dimensions
            if step % 3 == 0:
                batch['image'] = F.interpolate(batch['image'], size=(288, 224))
            elif step % 3 == 1:
                batch['image'] = F.interpolate(batch['image'], size=(224, 288))
            
            expert_loss = expert_trainer.train_step(batch)
            expert_losses.append(expert_loss)
            print(f"  Step {step+1}/20 - Expert Loss: {expert_loss:.4f}")
        
        # Plotting loss curves
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(router_losses)
        plt.title('Router Loss Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(expert_losses)
        plt.title('Expert Loss Curve')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.tight_layout()

        plot_path = "loss_curves.png"
        plt.savefig(plot_path)
        print(f"Loss curves plot saved to: {plot_path}")
        plt.close()

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
