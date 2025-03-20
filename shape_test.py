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
import time
import math
import numpy as np
import torch.nn.functional as F

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
from trainers.diffusion import DecentralizedFlowMatcher

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
        self.max_sampling_experts = 4
        
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
                
                # Use the mocked version
                class MockExpertDiT(ExpertDiT):
                    """Mocked version that disables distributed checks for testing"""
                    def __init__(self, config):
                        super().__init__(config)
                        
                    def forward(self, x, t, text_embeds=None):
                        # Safety measure - don't log so much in testing
                        orig_training = self.training
                        self.train(False)  # Temporarily disable training mode to avoid debug logs
                        
                        result = super().forward(x, t, text_embeds)
                        
                        # Restore original training state
                        self.train(orig_training)
                        return result
                
                self.expert = MockExpertDiT(config).to(device)
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
        
        # Track loss statistics
        loss_values = []
        for _ in range(10):
            loss = expert_trainer.train_step(batch)
            loss_values.append(loss)
            assert 0 < loss < 100, "Loss out of expected range"

        logger.info(f"Loss trajectory: {loss_values}")
        assert np.mean(loss_values[-3:]) < np.mean(loss_values[:3]), "Loss should decrease"
        
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

def test_sampling_pipeline(config=None, device="cuda", num_samples=4):
    """Test the sampling pipeline with simulated data and expert sharing"""
    logger.info("\n=== Testing DDM Sampling Pipeline with Expert Sharing ===")
    
    if config is None:
        config = DummyConfig()
        # Add sampling-specific config params
        config.max_sampling_experts = 4
        config.fast_validation = True
        config.sampling_steps = 20
    
    # Create a logger that will show detailed info
    sampling_logger = logging.getLogger("sampling_test")
    sampling_logger.setLevel(logging.DEBUG)
    
    # Define bucket dimensions and vae_scale_factor
    bucket_size = (512, 512)  # Square format from the config buckets
    vae_scale_factor = 8
    latent_channels = getattr(config, 'latent_channels', 16)
    
    # Calculate latent dimensions
    w, h = bucket_size
    latent_h, latent_w = h // vae_scale_factor, w // vae_scale_factor
    
    # Create input shape for sampling
    shape = (num_samples, latent_channels, latent_h, latent_w)
    sampling_logger.info(f"Testing sampling with shape: {shape}")
    
    # Initialize router model
    router_model = RouterModel(config).to(device)
    router_model.eval()
    sampling_logger.info("Router model initialized and set to eval mode")
    
    # Track timing
    start_time = time.time()
    
    # Create multiple expert models to simulate expert sharing
    num_experts = min(config.num_experts, config.max_sampling_experts)
    sampling_logger.info(f"Creating {num_experts} experts for testing expert sharing")
    
    experts_dict = {}
    for i in range(num_experts):
        try:
            expert_model = ExpertDiT(config).to(device)
            expert_model.eval()
            experts_dict[i] = expert_model
            sampling_logger.info(f"Expert {i} initialized and set to eval mode")
        except Exception as e:
            sampling_logger.error(f"Failed to create expert {i}: {str(e)}")
    
    sampling_logger.info(f"Created {len(experts_dict)} experts in {time.time() - start_time:.2f}s")
    
    # Test with optimized sampling approach
    try:
        # Set all experts to eval mode
        for expert in experts_dict.values():
            expert.eval()
        
        # Force router to eval mode
        router_model.eval()
        
        # Create initial noise
        latents = torch.randn(shape, device=device)
        sampling_logger.info(f"Created initial latents with shape {latents.shape}")
        
        # Use torch.no_grad for all test operations
        with torch.no_grad():
            # First verify router and expert compatibility with simple forward pass
            t_idx = 999  # Start from the end (pure noise)
            t_tensor = torch.tensor([t_idx], device=device)
            
            # Test router with batch
            router_logits = router_model(latents[:1], t_tensor)
            sampling_logger.info(f"Router output shape: {router_logits.shape}")
            
            # Ensure router produces probabilities that sum to 1 with softmax
            router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
            sampling_logger.info(f"Router probabilities sum: {router_probs.sum(dim=1)}")
            
            # Test top-k selection
            top_k = min(2, len(experts_dict)) 
            weights, indices = torch.topk(router_probs, top_k, dim=-1)
            sampling_logger.info(f"Top-{top_k} expert indices: {indices}")
            sampling_logger.info(f"Top-{top_k} expert weights: {weights}")
            
            # Test expert outputs with the same input
            for expert_idx, expert in experts_dict.items():
                if expert_idx in indices:
                    sampling_logger.info(f"Testing expert {expert_idx}")
                    expert_output = expert(latents[:1], t_tensor)
                    if expert_output.shape == latents[:1].shape:
                        sampling_logger.info(f"✓ Expert {expert_idx} output shape matches: {expert_output.shape}")
                    else:
                        sampling_logger.error(f"❌ Expert {expert_idx} shape mismatch: {expert_output.shape} ≠ {latents[:1].shape}")
            
            # Now simulate a full sampling step (combining predictions)
            combined_pred = torch.zeros_like(latents[:1])
            for i, idx in enumerate(indices[0]):
                if idx.item() in experts_dict:
                    expert = experts_dict[idx.item()]
                    expert_pred = expert(latents[:1], t_tensor)
                    weight = weights[0, i].view(-1, 1, 1, 1)
                    combined_pred += weight * expert_pred
                    
            sampling_logger.info(f"Combined prediction shape: {combined_pred.shape}")
            
            # Test full sampling with streamlined steps
            steps = 10  # Use fewer steps for test
            sampling_logger.info(f"Simulating {steps} sampling steps with expert sharing")
            
            # Setup for sampling
            from trainers.diffusion import get_alphas_and_betas, ddim_step
            
            alphas, alpha_bar, betas = get_alphas_and_betas(steps, "cosine")
            alphas = alphas.to(device)
            alpha_bar = alpha_bar.to(device)
            betas = betas.to(device)
            
            # Reset latents
            latents = torch.randn(shape, device=device)
            
            # Start timing full sampling
            sampling_start = time.time()
            
            # Simulate the full sampling process
            for i in range(steps):
                # Current timestep
                t_idx = steps - i - 1
                t_tensor = torch.tensor([t_idx], device=device)
                
                # Get router predictions
                router_logits = router_model(latents, t_tensor.repeat(latents.shape[0]))
                router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
                
                # Get top-k experts
                weights, indices = torch.topk(router_probs, top_k, dim=-1)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                
                # Get unique experts needed
                unique_experts = torch.unique(indices).cpu().tolist()
                sampling_logger.info(f"Step {i}: Using experts {unique_experts}")
                
                # Run expert predictions
                expert_predictions = {}
                for expert_idx in unique_experts:
                    if expert_idx in experts_dict:
                        expert = experts_dict[expert_idx]
                        pred = expert(latents, t_tensor.repeat(latents.shape[0]))
                        expert_predictions[expert_idx] = pred
                
                # Combine expert outputs
                combined_pred = torch.zeros_like(latents)
                for batch_idx in range(latents.shape[0]):
                    for i, expert_idx in enumerate(indices[batch_idx]):
                        expert_idx = expert_idx.item()
                        if expert_idx in expert_predictions:
                            # Add weighted prediction for this batch item
                            weight = weights[batch_idx, i].item()
                            combined_pred[batch_idx] += weight * expert_predictions[expert_idx][batch_idx]
                
                # Apply the ddim step to update latents
                next_t = torch.full_like(t_tensor, t_idx-1, device=device) 
                if t_idx > 0:
                    latents = ddim_step(
                        lambda x_t, t, c: combined_pred,  # Use precomputed prediction
                        latents,
                        t_tensor.repeat(latents.shape[0]),
                        next_t.repeat(latents.shape[0]),
                        alphas,
                        alpha_bar,
                        eta=0.0
                    )
            
            sampling_time = time.time() - sampling_start
            sampling_logger.info(f"Completed {steps} sampling steps in {sampling_time:.2f}s ({steps/sampling_time:.2f} steps/s)")
            sampling_logger.info(f"Final latents shape: {latents.shape}")
            
            # Memory usage info
            if torch.cuda.is_available():
                sampling_logger.info(f"GPU memory: {torch.cuda.memory_allocated(device)/1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device)/1e9:.2f} GB reserved")
            
            sampling_logger.info("✓ DDM sampling simulation completed successfully!")
            
    except Exception as e:
        sampling_logger.error(f"Error in sampling test: {str(e)}")
        import traceback
        sampling_logger.error(traceback.format_exc())
    
    # Clean up
    try:
        for expert in experts_dict.values():
            del expert
        del router_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
        
    return

def test_flow_matching_loss(config=None, device="cuda"):
    """Validate flow matching target and loss calculations"""
    logger.info("\n=== Testing Flow Matching Calculations ===")
    
    if config is None:
        config = DummyConfig()
    
    # Initialize flow matcher with different loss types
    loss_types = ['mse', 'huber', 'l1']
    
    for loss_type in loss_types:
        logger.info(f"\nTesting {loss_type.upper()} loss:")
        
        # Create flow matcher
        fm = DecentralizedFlowMatcher(loss_type=loss_type)
        
        # Create dummy data
        x0 = torch.randn(2, config.latent_channels, 32, 32, device=device)
        t = torch.tensor([0.1, 0.9], device=device)
        noise = torch.randn_like(x0)
        
        # Compute xt
        alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
        sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
        xt = alpha_t * x0 + sigma_t * noise
        
        # Calculate target
        target = fm.compute_flow_matching_target(x0, xt, t)
        logger.info(f"Target shape: {target.shape}")
        
        # Validate target properties
        assert not torch.isnan(target).any(), "NaNs in flow target"
        assert target.requires_grad == False, "Target should not require grad"
        
        # Create dummy predictions
        pred = torch.randn_like(target)
        
        # Calculate loss
        loss = fm.compute_flow_matching_loss(pred, target)
        logger.info(f"Loss value: {loss.item():.4f}")
        
        # Basic loss validation
        assert loss > 0, "Loss should be positive"
        assert loss.requires_grad == False, "Loss should not require grad"

        # Validate against known values
        x0 = torch.zeros(1, config.latent_channels, 4, 4, device=device)
        t = torch.tensor([0.0], device=device)
        xt = x0.clone()
        target = fm.compute_flow_matching_target(x0, xt, t)
        assert torch.allclose(target, torch.zeros_like(target)), "t=0 target should be zero"

        x0 = torch.ones(1, 1, 1, 1, device=device)
        t = torch.tensor([1.0], device=device)
        xt = torch.zeros(1, 1, 1, 1, device=device)
        target = fm.compute_flow_matching_target(x0, xt, t)
        assert torch.allclose(target, (x0 - xt)/1.0), "t=1 target should be (x0 - xt)"

def test_temperature_annealing():
    """Validate router temperature annealing"""
    logger.info("\n=== Testing Router Temperature Annealing ===")
    
    fm = DecentralizedFlowMatcher()
    initial_temp = fm.temperature
    
    # Simulate training steps
    for step in range(1000):
        fm.temperature = max(0.5, fm.temperature * (1 - fm.temp_anneal_rate))
    
    logger.info(f"Initial temp: {initial_temp:.2f}, Final temp: {fm.temperature:.2f}")
    assert fm.temperature >= 0.5, "Temperature should not drop below 0.5"
    assert fm.temperature < initial_temp, "Temperature should decrease"

def enhanced_test_sampling(config=None, device="cuda", num_samples=4):
    """Enhanced sampling test with validation checks"""
    # ... [existing setup code] ...
    
    # Add validation checks
    sampling_logger.info("Adding validation checks:")
    
    # Check 1: Verify expert combination weights sum to 1
    weight_sum = weights.sum(dim=-1)
    sampling_logger.info(f"Expert weight sums: {weight_sum}")
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), "Weights should sum to 1"
    
    # Check 2: Validate latent dimensions through sampling steps
    for i in range(steps):
        # ... [existing sampling code] ...
        
        # Validate latent stats
        current_mean = latents.mean().item()
        current_std = latents.std().item()
        sampling_logger.info(f"Step {i}: Latent mean={current_mean:.4f}, std={current_std:.4f}")
        
        assert not torch.isnan(latents).any(), "NaNs in latents"
        assert abs(current_mean) < 2.0, "Latent mean out of expected range"
        assert 0.5 < current_std < 2.0, "Latent std out of expected range"

def test_router_distribution(config=None, device="cuda"):
    """Validate router output distribution properties"""
    # ... [setup code] ...
    
    # Test 1: Uniform input
    uniform_input = torch.randn(1, config.latent_channels, 8, 8, device=device)
    logits = model(uniform_input, torch.tensor([0.5], device=device))
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0)), "Probabilities should sum to 1"
    
    # Test 2: Extreme temperatures
    model.temperature = 0.1
    sharp_probs = F.softmax(logits/model.temperature, dim=-1)
    assert sharp_probs.max() > 0.9, "Low temp should sharpen distribution"

def main():
    parser = argparse.ArgumentParser(description="Test shapes in Decentralized Diffusion Models")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA")
    parser.add_argument("--expert", action="store_true", help="Test expert model shapes")
    parser.add_argument("--router", action="store_true", help="Test router model shapes")
    parser.add_argument("--train", action="store_true", help="Test training step")
    parser.add_argument("--sampling", action="store_true", help="Test sampling pipeline")
    parser.add_argument("--flow", action="store_true", help="Test flow matching calculations")
    parser.add_argument("--temp", action="store_true", help="Test temperature annealing")
    args = parser.parse_args()
    
    # Default to testing everything if no specific test is selected
    if not (args.expert or args.router or args.train or args.sampling or args.flow or args.temp):
        args.expert = args.router = args.train = args.sampling = args.flow = args.temp = True
    
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
    
    # Run sampling test
    if args.sampling:
        test_sampling_pipeline(config, device, num_samples=4)
    
    # Add new test options
    if args.flow:
        test_flow_matching_loss(config, device)
    if args.temp:
        test_temperature_annealing()
    
    logger.info("Shape tests completed")

if __name__ == "__main__":
    main()

class MockExpertDiT(ExpertDiT):
    """Mocked version that disables distributed checks for testing"""
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x, t, text_embeds=None):
        # Safety measure - don't log so much in testing
        orig_training = self.training
        self.train(False)  # Temporarily disable training mode to avoid debug logs
        
        result = super().forward(x, t, text_embeds)
        
        # Restore original training state
        self.train(orig_training)
        return result 