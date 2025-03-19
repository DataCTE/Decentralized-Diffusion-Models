#!/usr/bin/env python3
"""
Shape test script for Decentralized Diffusion Models

This script creates dummy inputs and models to verify tensor shapes
throughout the pipeline, helping diagnose dimension mismatch issues.
"""

import torch
import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("shape_test")

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit import ExpertDiT
from models.router import RouterModel
from trainers.expert import ExpertTrainer
from trainers.router import RouterTrainer
from config import get_config
from utils.checkpoint import load_model_checkpoint

class DummyConfig:
    """Dummy configuration with minimal settings for shape testing"""
    def __init__(self):
        # Model settings
        self.hidden_dim = 768
        self.patch_size = 16
        self.latent_channels = 16
        self.num_heads = 12
        self.num_layers = 12
        self.ffn_dim = 3072
        self.num_experts = 8
        
        # Router settings
        self.router_hidden_size = 512
        self.router_num_heads = 8
        
        # Training settings
        self.learning_rate = 1e-4
        self.adam_betas = (0.9, 0.999)
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.use_mixed_precision = False
        self.num_steps = 100000
        self.use_gradient_checkpointing = False
        
        # Flow matching parameters
        self.sigma = 0.5
        self.loss_type = 'huber'
        
        # FSDP parameters
        self.fsdp_sharding_strategy = "FULL_SHARD"
        self.fsdp_cpu_offload = False
        self.fsdp_backward_prefetch = "BACKWARD_PRE"
        self.fsdp_auto_wrap_policy = "LAMBDA"
        self.fsdp_min_num_params = 1e6
        self.fsdp_use_orig_params = True
        self.fsdp_limit_all_gathers = True
        self.router_learning_rate = 1e-4

def create_dummy_batch(batch_size=1, device="cuda", image_size=(576, 448)):
    """Create dummy batch data for testing"""
    height, width = image_size
    
    # Create dummy RGB images
    images = torch.randn(batch_size, 3, height, width, device=device)
    
    # Create dummy text captions
    captions = ["A test image"] * batch_size
    
    return {
        "image": images,
        "caption": captions,
    }

def test_expert_shapes(config, device="cuda"):
    """Test shapes through the expert model pipeline"""
    logger.info("=== Testing Expert DiT Model Shapes ===")
    
    # Initialize expert model directly (not wrapped in FSDP)
    model = ExpertDiT(config).to(device)
    model.eval()
    
    # Create a batch with different standard image sizes
    test_sizes = [(576, 448), (512, 512), (448, 576), (640, 384), (384, 640)]
    
    for size in test_sizes:
        logger.info(f"\nTesting image size: {size}")
        batch = create_dummy_batch(batch_size=1, device=device, image_size=size)
        
        # Simulate latent encoding (normally done by VAE)
        # Assuming 16x downsampling from image to latent space
        h, w = size
        latent_h, latent_w = h // 8, w // 8
        latents = torch.randn(1, config.latent_channels, latent_h, latent_w, device=device)
        
        # Timesteps
        t = torch.randint(0, 1000, (1,), device=device)
        
        # Text embeddings (simulating CLIP output)
        text_embeds = torch.randn(1, 77, 768, device=device)
        
        # Print input shapes
        logger.info(f"Input latent shape: {latents.shape}")
        logger.info(f"Timestep shape: {t.shape}")
        logger.info(f"Text embedding shape: {text_embeds.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(latents, t, text_embeds)
            
        # Print output shape
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Target shape should match input: {latents.shape}")
        
        if output.shape != latents.shape:
            logger.warning(f"❌ Shape mismatch: {output.shape} != {latents.shape}")
        else:
            logger.info(f"✓ Shapes match correctly")

def test_router_shapes(config, device="cuda"):
    """Test shapes through the router model pipeline"""
    logger.info("\n=== Testing Router Model Shapes ===")
    
    # Initialize router model directly (not wrapped in FSDP)
    model = RouterModel(config).to(device)
    model.eval()
    
    # Create a batch with different standard image sizes
    test_sizes = [(576, 448), (512, 512), (448, 576), (640, 384), (384, 640)]
    
    for size in test_sizes:
        logger.info(f"\nTesting image size: {size}")
        batch = create_dummy_batch(batch_size=1, device=device, image_size=size)
        
        # Simulate latent encoding (normally done by VAE)
        h, w = size
        latent_h, latent_w = h // 8, w // 8
        latents = torch.randn(1, config.latent_channels, latent_h, latent_w, device=device)
        
        # Timesteps - FIXED: Convert to float
        t = torch.randint(0, 1000, (1,), device=device).float()
        
        # Print input shapes
        logger.info(f"Input latent shape: {latents.shape}")
        logger.info(f"Timestep shape: {t.shape}, dtype: {t.dtype}")
        
        # Forward pass
        with torch.no_grad():
            output = model(latents, t)
            
        # Print output shape
        logger.info(f"Router output logits shape: {output.shape}")
        logger.info(f"Expected shape: [1, {config.num_experts}]")
        
        if output.shape != torch.Size([1, config.num_experts]):
            logger.warning(f"❌ Shape mismatch: {output.shape} != [1, {config.num_experts}]")
        else:
            logger.info(f"✓ Shapes match correctly")
            
def test_training_step(device="cuda"):
    """Test the full training step with dummy data"""
    logger.info("\n=== Testing Training Step ===")
    
    # Create config
    config = DummyConfig()
    
    # Create dummy batch
    batch = create_dummy_batch(batch_size=1, device=device)
    
    # Test expert training step (create standalone trainer without FSDP)
    logger.info("\nTesting Expert Training Step")
    try:
        # Create a standalone test trainer that doesn't inherit from ExpertTrainer
        class TestExpertTrainer:
            def __init__(self, config, device):
                self.config = config
                self.device = device
                self.expert_idx = 0
                self.rank = 0
                self.expert = ExpertDiT(config).to(device)
                self.optimizer = torch.optim.AdamW(
                    self.expert.parameters(),
                    lr=config.learning_rate,
                    betas=config.adam_betas,
                    weight_decay=config.weight_decay
                )
                
                # Create flow matcher
                from trainers.diffusion import DecentralizedFlowMatcher
                self.flow_matcher = DecentralizedFlowMatcher(
                    sigma=config.sigma, 
                    loss_type=config.loss_type
                )
                
                # Mock VAE and CLIP
                class MockEncoder:
                    def __init__(self, device, channels):
                        self.device = device
                        self.channels = channels
                    def encode(self, x):
                        # Create dummy latents with 1/8 spatial dims
                        h, w = x.shape[2] // 8, x.shape[3] // 8
                        return torch.randn(x.shape[0], self.channels, h, w, device=self.device)
                
                class MockCLIP:
                    def __init__(self, device):
                        self.device = device
                    def encode(self, text):
                        # Create dummy CLIP embeddings
                        return torch.randn(len(text), 77, 768, device=self.device)
                
                self.vae = MockEncoder(device, config.latent_channels)
                self.clip = MockCLIP(device)
                
                # Other required attributes
                self.alphas = torch.linspace(1.0, 0.02, 1000, device=device)
                self.alpha_bar = torch.cumprod(self.alphas, dim=0)
                
                # Mock scheduler
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lambda x: 1.0
                )
            
            # Copy the train_step method from ExpertTrainer but simplify it
            def train_step(self, batch):
                images = batch["image"].to(self.device)
                
                # Use mixed precision training if configured
                scaler = torch.amp.GradScaler('cuda', enabled=False)  # Simplified for testing
                
                # VAE encoding
                latents = self.vae.encode(images)
                
                # Sample random timesteps
                t_indices = torch.randint(0, 1000, (latents.size(0),), device=self.device)
                t = t_indices.float() / 1000.0
                
                # Sample random noise
                noise = torch.randn_like(latents)
                
                # Forward process
                alpha_t = torch.cos((t + 0.008)/1.008 * torch.pi/2).pow(2)[:,None,None,None]
                sigma_t = torch.sin(t * torch.pi/2)[:,None,None,None]
                latent_t = alpha_t * latents + sigma_t * noise
                
                # Text conditioning
                text_embeds = self.clip.encode(batch["caption"])
                
                # Expert prediction
                pred_flow = self.expert(latent_t, t_indices, text_embeds)
                
                # The target flow field
                target_flow = self.flow_matcher.compute_flow_matching_target(
                    latents, latent_t, t
                )
                
                # Flow matching loss
                loss = self.flow_matcher.compute_flow_matching_loss(
                    pred_flow, target_flow
                )
                
                # Simplified optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                return loss.item()

        expert_trainer = TestExpertTrainer(config, device)
        loss = expert_trainer.train_step(batch)
        logger.info(f"Expert training step completed with loss: {loss}")
        
    except Exception as e:
        logger.error(f"Error in expert training step: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test router training step
    logger.info("\nTesting Router Training Step")
    try:
        # Create modified trainer that doesn't use FSDP
        class TestRouterTrainer(RouterTrainer):
            def __init__(self, config, device):
                # Initialize with minimal requirements
                self.config = config
                self.device = device
                self.rank = 0
                self.router = RouterModel(config).to(device)
                self.optimizer = torch.optim.AdamW(
                    self.router.parameters(),
                    lr=config.router_learning_rate,
                    weight_decay=config.weight_decay
                )
                self.criterion = torch.nn.CrossEntropyLoss()
                
                # Mock VAE
                class MockEncoder:
                    def __init__(self, device, channels):
                        self.device = device
                        self.channels = channels
                    def encode(self, x):
                        # Create dummy latents with 1/8 spatial dims
                        h, w = x.shape[2] // 8, x.shape[3] // 8
                        return torch.randn(x.shape[0], self.channels, h, w, device=self.device)
                
                self.vae = MockEncoder(device, config.latent_channels)
                
                # Mock scheduler
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lambda x: 1.0
                )
        
        router_trainer = TestRouterTrainer(config, device)
        
        # Create a mock batch that ensures t_indices is float
        mock_batch = create_dummy_batch(batch_size=1, device=device)
        
        # When we execute train_step, it will now use float timesteps
        loss = router_trainer.train_step(mock_batch)
        logger.info(f"Router training step completed with loss: {loss}")
        
    except Exception as e:
        logger.error(f"Error in router training step: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Test shapes in Decentralized Diffusion Models")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA")
    parser.add_argument("--expert", action="store_true", help="Test expert model shapes")
    parser.add_argument("--router", action="store_true", help="Test router model shapes")
    parser.add_argument("--train", action="store_true", help="Test training step")
    args = parser.parse_args()
    
    # Default to testing everything if no specific test is selected
    if not (args.expert or args.router or args.train):
        args.expert = args.router = args.train = True
    
    # Set device
    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Running shape tests on {device}")
    
    # Create config
    config = DummyConfig()
    
    # Run tests
    if args.expert:
        test_expert_shapes(config, device)
    
    if args.router:
        test_router_shapes(config, device)
    
    if args.train:
        test_training_step(device)
    
    logger.info("Shape tests completed")

if __name__ == "__main__":
    main() 