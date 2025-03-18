"""Coordinator for Decentralized Diffusion Models with Uniform Distribution"""

import os
import torch
import datetime
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import threading
from tqdm.auto import tqdm
import concurrent.futures

# Import needed components
from trainers.router import RouterTrainer
from trainers.sampling import ddm_sample
from trainers.diffusion import DecentralizedFlowMatcher
from data.dataset import DDMDataset
from utils.logging import setup_logger
from utils.checkpoint import save_coordinator_checkpoint, load_coordinator_checkpoint

# Setup logger
logger = setup_logger("DDMCoordinator")

# Direct console print function for immediate feedback
def debug_print(message, rank=None, force=False):
    """Print directly to console regardless of logger configuration"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if rank is not None:
        prefix = f"[DDMT-{rank}]"
    else:
        prefix = "[DDMT]"
        
    if force or (rank is not None and rank == 0) or rank is None:
        print(f"{prefix} [{timestamp}] {message}", flush=True)

class DDMTrainingCoordinator:
    """Coordinator for Decentralized Diffusion Models with uniform data distribution"""
    
    def __init__(self, config, rank, world_size, cache_manager=None, progress_callback=None):
        """
        Initialize coordinator for decentralized diffusion
        
        Args:
            config: Configuration object
            rank: Process rank (0 is main)
            world_size: Total number of processes
            cache_manager: Optional cache manager
            progress_callback: Optional callback function to report initialization progress
        """
        init_start_time = time.time()
        debug_print(f"Starting DDM initialization on rank {rank}/{world_size}", rank, force=True)
        
        # Store basic configuration
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.progress_callback = progress_callback
        self.cache_manager = cache_manager
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        # Parallel initialization components
        self._init_parallel_components()
        
        # Defer non-critical initialization
        self.flow_matcher = None  # Will be created on first training step
        
        # Final initialization sync
        total_init_time = time.time() - init_start_time
        debug_print(f"DDM initialization completed in {total_init_time:.2f}s", rank, force=True)
    
    def _init_parallel_components(self):
        """Initialize critical components with async dataset loading"""
        pbar = None
        if self.rank == 0:
            pbar = tqdm(
                total=2,  # Reduced from 3 to 2 (router and experts only)
                desc="Initializing Components",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar:20}{r_bar}"
            )

        # Use ThreadPoolExecutor for better resource management
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit data loading separately without tracking
            data_future = executor.submit(self._init_data_loaders)
            
            # Only track these two components in progress bar
            futures = {
                executor.submit(self._init_router): "router",
                executor.submit(self._init_expert_indices): "experts"
            }

            try:
                # Process completion with progress updates
                for future in concurrent.futures.as_completed(futures):
                    component = futures[future]
                    future.result()  # Raise exceptions if any
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed: {component}")
            finally:
                if pbar is not None:
                    pbar.close()
            
            # Ensure data loading completes before continuing
            data_future.result()
    
    def _init_data_loaders(self):
        """Initialize data loaders with uniform distribution"""
        debug_print(f"Creating data loaders on rank {self.rank}", self.rank, force=True)
        
        # Initialize the dataset
        train_dataset = DDMDataset(
            config=self.config,
            split='train'
        )
        
        # Create validation dataset
        val_dataset = DDMDataset(
            config=self.config,
            split='val'
        )
        
        # Create distributed samplers if in distributed training
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers if hasattr(self.config, 'num_workers') else 4,
            sampler=val_sampler,
            pin_memory=True,
            drop_last=False
        )
        
        debug_print(f"Created data loaders with {len(train_dataset)} training samples on rank {self.rank}", self.rank, force=True)
    
    def _init_expert_indices(self):
        """Determine expert assignments without model creation"""
        self.expert_indices = [
            idx for idx in range(self.config.num_experts)
            if idx % self.world_size == self.rank
        ]
        logger.info(f"Rank {self.rank} will manage {len(self.expert_indices)} experts")
    
    def _init_router(self):
        """Initialize router with async FSDP wrapping"""
        logger.info(f"Creating router on rank {self.rank}")
        self.router = RouterTrainer(
            config=self.config,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
    
    def train(self, num_steps=None):
        """Run training for specified number of steps, following DDM approach"""
        if num_steps is None:
            num_steps = self.config.num_steps
            
        logger.info(f"Starting DDM training for {num_steps} steps on rank {self.rank}")
        
        # Initialize flow matcher on first use
        if self.flow_matcher is None:
            self.flow_matcher = DecentralizedFlowMatcher(
                sigma=getattr(self.config, 'sigma', 0.5),
                loss_type=getattr(self.config, 'loss_type', 'huber')
            )
        
        # Create train dataloader iterator
        train_iter = iter(self.train_loader)
        
        # Train loop implementing the DDM training approach
        for step in range(num_steps):
            batch = next(train_iter)
            
            # Joint training of experts and router
            expert_loss = self.train_experts(batch)  # Updates experts
            router_loss = self.train_router(batch)   # Updates router
            
            # Log every N steps
            if step % 100 == 0 or step == num_steps - 1:
                logger.info(f"Step {step}/{num_steps}: Expert loss = {expert_loss:.4f}, Router loss = {router_loss:.4f}")
                
            # Periodic validation
            if step % 1000 == 0:
                self.validate(step)
                
            # Save checkpoint every N steps
            if step > 0 and step % 5000 == 0:
                self.save_checkpoint(step)
    
    def train_experts(self, batch):
        """Train expert models using the DDM approach"""
        for expert_idx in self.expert_indices:
            # Get expert from cache manager
            expert = self.cache_manager.get_expert(expert_idx)
            expert.train_step(batch)
    
    def train_router(self, batch):
        """Train router model using expert assignments as supervision"""
        # Distributed router training
        self.router.train_step(batch)
    
    def validate(self, step):
        """Run validation using DDM inference process"""
        # Only run validation on rank 0
        if self.rank != 0:
            return
            
        logger.info(f"Running validation at step {step}")
        
        # Generate samples using DDM sampling
        self.generate_samples(num_samples=4, step=step)
    
    def generate_samples(self, num_samples=4, step=None, prompts=None):
        """Generate samples using the DDM inference approach"""
        if self.rank != 0:
            return
            
        logger.info(f"Generating {num_samples} samples")
        
        # Create sample directory
        if step is not None:
            sample_dir = os.path.join(self.config.output_dir, 'samples', f'step_{step}')
        else:
            sample_dir = os.path.join(self.config.output_dir, 'samples')
            
        os.makedirs(sample_dir, exist_ok=True)
        
        # Collect all experts for sampling
        experts_dict = {expert_idx: self.cache_manager.get_expert(expert_idx) for expert_idx in self.expert_indices}
        
        # Use proper DDM sampling from trainers/sampling.py
        try:
            # Use the first bucket's dimensions for sampling
            # In real applications, you might want to sample from different buckets
            if hasattr(self.config, 'buckets') and len(self.config.buckets) > 0:
                w, h = self.config.buckets[0]  # Get dimensions from first bucket
                C = self.config.image_size[0]  # Get channel count
                shape = (num_samples, C, h, w)
                logger.info(f"Generating samples with dimensions {shape} from bucket 0")
            else:
                # Fallback to image_size
                C, H, W = self.config.image_size
                shape = (num_samples, C, H, W)
                logger.info(f"Generating samples with dimensions {shape} from image_size")
            
            # Get optional text embeddings if conditional
            text_embeddings = None
            uncond_embeddings = None
            if prompts is not None and hasattr(self, 'text_encoder') and self.text_encoder is not None:
                text_embeddings = []
                for prompt in prompts:
                    text_embeddings.append(self.text_encoder.encode(prompt))
                text_embeddings = torch.cat(text_embeddings, dim=0).to(self.device)
                
                # Create unconditional embeddings (empty string) for classifier-free guidance
                uncond_embeddings = self.text_encoder.encode([""] * num_samples).to(self.device)
            
            # Use ddm_sample from trainers/sampling.py for proper DDM sampling
            samples = ddm_sample(
                router=self.router,
                experts=experts_dict,
                shape=shape,
                steps=getattr(self.config, 'sampling_steps', 50),
                top_k=getattr(self.config, 'top_k', 1),
                device=self.device,
                cfg_scale=getattr(self.config, 'cfg_scale', 7.5),
                text_embeddings=text_embeddings,
                uncond_embeddings=uncond_embeddings,
                eta=getattr(self.config, 'eta', 0.0),
                scheduler=getattr(self.config, 'beta_schedule', "cosine"),
                verbose=True,
                temperature=getattr(self.config, 'temperature', 1.0)
            )
            
            # Save samples
            from torchvision.utils import save_image
            for i in range(num_samples):
                save_image(samples[i], os.path.join(sample_dir, f'sample_{i}.png'))
                    
            logger.info(f"Saved {num_samples} samples to {sample_dir}")
        except Exception as e:
            logger.error(f"Error generating samples: {e}")
    
    def save_checkpoint(self, step):
        """Save checkpoint of all components"""
        # Only save from main process unless configured otherwise
        if self.rank != 0 and not getattr(self.config, 'save_from_all_ranks', False):
            return
            
        logger.info(f"Saving checkpoint at step {step}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints', f'step_{step}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save router using its save_checkpoint method
        self.router.save_checkpoint(checkpoint_dir, step)
            
        # Save coordinator state
        save_coordinator_checkpoint(
            checkpoint_dir, 
            {
                "step": step,
                "config": self.config,
            }
        )
                    
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """Load checkpoint of all components"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load coordinator state
        coordinator_state = load_coordinator_checkpoint(checkpoint_dir)
        step = coordinator_state.get("step", 0) if coordinator_state else 0
        
        # Load router using its load_checkpoint method
        if self.router is not None:
            router_path = os.path.join(checkpoint_dir, 'router.pt')
            if os.path.exists(router_path):
                self.router.load_checkpoint(router_path)
                    
        return step
