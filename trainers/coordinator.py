"""Training coordinator for Decentralized Diffusion Models."""

import os
import torch
import torch.distributed
import logging
import math
from tqdm import tqdm
import wandb
from bitsandbytes.optim import AdamW8bit
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
import time
import torch.nn.functional as F
import threading  # Import threading at the top level

from data.dataset import DDMDataset, FeatureDataset, create_expert_bucket_loaders
from data.clustering import ClusterManager
from trainers.expert import ExpertTrainer, ExpertDiT
from trainers.router import RouterTrainer
from utils.vae import VAEWrapper
from utils.clip import CLIPTextEncoder
from inference import ddm_sample
from config import DDMConfig

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class DDMTrainingCoordinator:
    """Coordinates the training of DDM experts and router"""
    def __init__(self, config, rank, world_size):
        """
        Initialize the DDM Training Coordinator.
        
        Args:
            config: Configuration object
            rank: Process rank
            world_size: Total number of processes
        """
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize step counter
        self.current_step = 0
        
        # Don't initialize dataset here - moved to init_data_loaders()
        
        # Initialize router trainer
        self.router_trainer = RouterTrainer(config, self.device, self.rank)
        
        # Initialize expert trainers
        self.expert_trainers = [
            ExpertTrainer(i, config, self.device, self.rank)
            for i in range(config.num_experts)
        ]
        
        # Initialize previous clusters for tracking reassignments
        # We'll set this after dataset initialization
        self.previous_clusters = None
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize components
        self.init_data_loaders()
        self.init_models()
        self.init_metrics()

    def init_data_loaders(self):
        # Only rank 0 validates the dataset, others wait
        if self.rank == 0:
            logger.info("Rank 0 validating dataset for all processes")
            self.full_dataset = DDMDataset(self.config.dataset_path)
            self.cluster_manager = self.perform_initial_clustering()
        else:
            # Wait for rank 0 to finish validation
            torch.distributed.barrier()
            logger.info(f"Rank {self.rank} loading pre-validated dataset")
            self.full_dataset = DDMDataset(self.config.dataset_path, validate=False)
            self.full_dataset.cluster_labels = np.zeros(1, dtype=np.int64)

        # Make sure all ranks are ready before proceeding
        torch.distributed.barrier()
        
        self.sync_cluster_labels()
        
        # Now initialize previous_clusters with correct size
        self.previous_clusters = np.zeros(len(self.full_dataset), dtype=np.int32)
        
        self.expert_loaders = create_expert_bucket_loaders(
            self.full_dataset, self.config, self.world_size, self.rank
        )

    def perform_initial_clustering(self):
        cluster_manager = ClusterManager()
        feature_dataset = FeatureDataset(self.config.dataset_path, self.config)
        feature_loader = DataLoader(
            feature_dataset, 
            batch_size=self.config.feature_batch_size,
            num_workers=self.config.feature_workers
        )
        features = cluster_manager.extract_features(feature_loader)
        cluster_labels = cluster_manager.cluster_dataset(features)
        self.full_dataset.cluster_labels = cluster_labels
        return cluster_manager

    def sync_cluster_labels(self):
        cluster_tensor = torch.from_numpy(self.full_dataset.cluster_labels).to(self.device)
        torch.distributed.broadcast(cluster_tensor, src=0)
        self.full_dataset.cluster_labels = cluster_tensor.cpu().numpy()

    def init_models(self):
        self.expert_trainers = [
            ExpertTrainer(i, self.config, self.device, self.rank)
            for i in range(self.config.num_experts)
        ]
        self.router_trainer = RouterTrainer(self.config, self.device, self.rank)

    def init_metrics(self):
        if self.rank == 0:
            # Initialize wandb with project name and config
            wandb.init(
                project="ddm-training",
                config=vars(self.config),
                settings=wandb.Settings(start_method="thread", _disable_stats=True)
            )
            
            # Log model architecture information
            model_info = {
                "num_experts": self.config.num_experts,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "ffn_dim": self.config.ffn_dim,
                "latent_channels": self.config.latent_channels,
                "patch_size": self.config.patch_size,
                "image_size": self.config.image_size,
            }
            
            # Log training hyperparameters
            training_info = {
                "learning_rate": self.config.learning_rate,
                "router_learning_rate": self.config.router_learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "num_steps": self.config.num_steps,
                "save_interval": self.config.save_interval,
                "recluster_interval": self.config.recluster_interval,
            }
            
            # Log to wandb
            wandb.config.update({"model_architecture": model_info})
            wandb.config.update({"training_hyperparameters": training_info})
            
            # Log system info
            self._log_to_wandb_async({"system/num_gpus": self.world_size})
            
            # Log learning rate schedules
            self.log_lr_schedules()
            
            # Log model architecture as a graph
            self.log_model_graph()
            
            # Log dataset statistics
            self.log_dataset_stats()

    def run_training_cycle(self):
        """
        Run the main training cycle for the DDM model.
        This includes training experts, router, and performing reclustering.
        """
        logger.info(f"Starting training cycle with rank {self.rank}")
        
        # Initialize metrics tracking
        self.init_metrics()
        
        # Initialize time tracking for metrics
        self.last_log_time = time.time()
        self.last_log_step = self.current_step
        
        # Main training loop
        while self.current_step < self.config.num_steps:
            # Train experts
            for step in range(self.config.expert_steps_per_router):
                # Train each expert on its assigned data
                for expert_idx, expert_trainer in enumerate(self.expert_trainers):
                    loss = expert_trainer.train_step(self.current_step)
                    self.log_metrics(expert_idx, loss, step)
                
                # Log training metrics every 10 steps
                if self.current_step % 10 == 0:
                    self.log_training_metrics()
                
                # Run validation at specified intervals
                if self.current_step % self.config.save_interval == 0:
                    self.run_validation()
                
                # Save checkpoints at specified intervals
                if self.current_step % self.config.save_interval == 0:
                    self.save_checkpoints()
                
                # Log expert specialization metrics at specified intervals
                if self.current_step % self.config.save_interval == 0:
                    self.log_expert_specialization()
                
                # Increment step counter
                self.current_step += 1
                
                # Check if we've reached the maximum number of steps
                if self.current_step >= self.config.num_steps:
                    break
            
            # Train router
            self.train_router()
            
            # Perform reclustering at specified intervals
            if self.current_step % self.config.recluster_interval == 0:
                self.perform_reclustering()
            
            # Perform distillation at specified intervals
            if self.config.distill and self.current_step % self.config.distill_interval == 0:
                self.perform_distillation()
        
        # Final checkpoint save
        self.save_checkpoints()
        
        # Final validation
        self.run_validation()
        
        # Final expert specialization metrics
        self.log_expert_specialization()
        
        logger.info(f"Training completed after {self.current_step} steps")

    def needs_reclustering(self):
        return (self.current_step > 0 and 
                self.current_step % self.config.recluster_interval == 0)
    
    def perform_periodic_reclustering(self):
        """Reclustering the dataset periodically as described in the paper"""
        if self.rank == 0:
            logger.info(f"Performing reclustering at step {self.current_step}")
            
            # Extract features from current dataset
            feature_dataset = FeatureDataset(self.config.dataset_path, self.config)
            
            # Use centralized config parameters for feature extraction
            feature_loader = DataLoader(
                feature_dataset, 
                batch_size=self.config.feature_batch_size,
                num_workers=self.config.feature_workers,
                pin_memory=self.config.pin_memory
            )
            
            # Use autocast to reduce memory usage during feature extraction
            with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                features = self.cluster_manager.extract_features(feature_loader)
            
            # Perform clustering
            new_cluster_labels = self.cluster_manager.cluster_dataset(features)
            self.full_dataset.cluster_labels = new_cluster_labels
        
        # Sync new cluster labels across all processes
        self.sync_cluster_labels()
        
        # Recreate expert loaders with new cluster assignments
        self.expert_loaders = create_expert_bucket_loaders(
            self.full_dataset, self.config, self.world_size, self.rank
        )

    def perform_reclustering(self):
        """
        Perform reclustering of the dataset based on the current router model.
        This reassigns data points to experts based on router predictions.
        """
        logger.info(f"Performing reclustering at step {self.current_step}")
        
        # Extract features from the dataset
        features = self.extract_features()
        
        # Cluster the features using the router
        clusters = self.cluster_features(features)
        
        # Create new dataloaders for each expert based on the clustering
        expert_loaders = create_expert_bucket_loaders(
            self.full_dataset, 
            clusters, 
            self.config.num_experts, 
            self.config.batch_size
        )
        
        # Assign the new dataloaders to the expert trainers
        for expert_idx, expert_trainer in enumerate(self.expert_trainers):
            if expert_idx < len(expert_loaders):
                expert_trainer.dataloader = expert_loaders[expert_idx]
                logger.info(f"Expert {expert_idx} assigned {len(expert_loaders[expert_idx].dataset)} samples")
            else:
                logger.warning(f"Expert {expert_idx} has no data after reclustering")
        
        # Log dataset statistics after reclustering
        self.log_dataset_stats()
        
        # Log cluster assignments
        self.log_cluster_assignments(clusters)
        
        logger.info("Reclustering completed")

    def train_experts(self):
        for expert_idx, trainer in enumerate(self.expert_trainers):
            loader = self.expert_loaders[expert_idx]
            progress = self.create_progress_bar(expert_idx)
            
            for step in range(self.config.save_interval):
                try:
                    batch = next(loader)
                    loss = trainer.train_step(batch)
                    
                    if self.rank == 0:
                        self.log_metrics(expert_idx, step, loss)
                        progress.update(1)
                        
                except Exception as e:
                    logger.error(f"Expert {expert_idx} error: {str(e)}")
                    continue

            if self.rank == 0:
                progress.close()

    def train_router(self):
        if self.rank == 0 and self.current_step % self.config.save_interval == 0:
            # Use the router batch size from config to ensure centralized configuration
            router_batch_size = self.config.router_batch_size
            avg_loss = self.router_trainer.train_epoch(
                DataLoader(
                    self.full_dataset, 
                    batch_size=router_batch_size, 
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    pin_memory=True
                )
            )
            logger.info(f"Router Training Loss: {avg_loss:.4f}")
            
            # Log router metrics to wandb using helper method
            log_data = {
                "router/train_loss": avg_loss,
                "step": self.current_step
            }
            self._log_to_wandb_async(log_data)
            
            # Sync router across devices
            router_state = [self.router_trainer.router.state_dict()]
            torch.distributed.broadcast_object_list(router_state, src=0)
        else:
            router_state = [None]
            torch.distributed.broadcast_object_list(router_state, src=0)
            self.router_trainer.router.load_state_dict(router_state[0])

    def run_validation(self):
        """
        Run validation by generating samples and logging them to wandb.
        This is called at regular intervals during training.
        """
        if self.rank == 0 and self.current_step % self.config.save_interval == 0:
            logger.info(f"Running validation at step {self.current_step}")
            
            # Set models to eval mode
            self.router_trainer.router.eval()
            for trainer in self.expert_trainers:
                trainer.expert.eval()
            
            # Generate validation prompts
            validation_prompts = [
                "a photo of a cat",
                "a photo of a dog",
                "a beautiful sunset over the ocean",
                "a futuristic cityscape at night"
            ]
            
            # Create encoder instances
            clip_encoder = CLIPTextEncoder(self.device, self.config)
            vae_wrapper = VAEWrapper(self.device, self.config)
            
            # Encode validation prompts
            with torch.no_grad():
                text_embeddings = clip_encoder.encode(validation_prompts)
                uncond_embeddings = clip_encoder.encode([""] * len(validation_prompts))
            
            # Generate samples
            with torch.no_grad():
                samples = ddm_sample(
                    self.config,
                    self.router_trainer.router,
                    [trainer.expert for trainer in self.expert_trainers],
                    vae_wrapper,
                    text_embeddings,
                    num_steps=self.config.inference_steps,
                    batch_size=len(validation_prompts),
                    device=self.device,
                    guidance_scale=self.config.cfg_scale,
                    uncond_embeddings=uncond_embeddings,
                    log_to_wandb=True
                )
            
            # Log samples to wandb
            if wandb.run is not None:
                # Convert samples from [-1, 1] to [0, 1] for wandb
                samples = (samples + 1) / 2
                
                # Create a grid of images
                samples_grid = make_grid(samples, nrow=2)
                samples_grid_np = samples_grid.permute(1, 2, 0).cpu().numpy()
                
                # Prepare grid data for logging
                grid_data = {
                    "validation/samples_grid": wandb.Image(samples_grid_np, caption=f"Validation samples at step {self.current_step}"),
                    "step": self.current_step
                }
                
                # Log grid using helper method
                self._log_to_wandb_async(grid_data)
                
                # Prepare individual samples data
                for i, (sample, prompt) in enumerate(zip(samples, validation_prompts)):
                    sample_np = sample.permute(1, 2, 0).cpu().numpy()
                    sample_data = {
                        f"validation/sample_{i}": wandb.Image(sample_np, caption=f"{prompt} (step {self.current_step})"),
                        "step": self.current_step
                    }
                    
                    # Log each sample using helper method
                    self._log_to_wandb_async(sample_data)
            
            # Set models back to train mode
            self.router_trainer.router.train()
            for trainer in self.expert_trainers:
                trainer.expert.train()
            
            logger.info(f"Validation completed at step {self.current_step}")

    def ddm_sample(self, router, experts, vae_wrapper, clip_encoder, text_embeddings, num_steps=50, batch_size=1, device=None, guidance_scale=7.5, uncond_embeddings=None, log_to_wandb=False):
        """
        Sample from the DDM ensemble using flow matching according to paper Section 3.4
        
        Args:
            router: Router model
            experts: List of expert models
            vae_wrapper: VAE for encoding/decoding
            clip_encoder: CLIP text encoder
            text_embeddings: Text embeddings for conditioning
            num_steps: Number of sampling steps
            batch_size: Batch size for sampling
            device: Device for computation
            guidance_scale: Guidance scale for guidance
            uncond_embeddings: Unconditional embeddings for guidance
            log_to_wandb: Whether to log to wandb
        """
        if device is None:
            device = router.device
        
        # Paper-recommended timestep schedule (cosine) from Section 3.4
        # The paper uses a cosine schedule for alpha_t and sigma_t
        ts = torch.linspace(0, 1, num_steps, device=device)
        alphas = torch.cos(ts * math.pi / 2)
        sigmas = torch.sin(ts * math.pi / 2)
        
        # Initialize latent with paper-specified noise scale (Section 3.4)
        # The paper initializes with Gaussian noise scaled by sigma
        latent = torch.randn(
            batch_size, 
            self.config.latent_channels, 
            self.config.image_size // self.config.patch_size, 
            self.config.image_size // self.config.patch_size, 
            device=device
        ) * self.config.sigma
        
        # Text conditioning
        if text_embeddings is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                text_embeddings = clip_encoder.encode(text_embeddings)
        
        # Create a mapping from expert model to expert index for logging
        expert_indices = {expert: i for i, expert in enumerate(experts)}
        
        # Flow matching sampling loop (Algorithm 2 in the paper)
        for i in tqdm(range(num_steps-1), desc="Sampling"):
            t = ts[i]
            t_tensor = torch.tensor([t], device=device)
            alpha_t = alphas[i]
            sigma_t = sigmas[i]
            
            with torch.no_grad():
                # Get router probabilities (Equation 5 from paper)
                # p_k(x_t, t) = softmax(router(x_t, t))
                router_logits = router(latent, t_tensor)
                router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
                
                if self.config.use_top_k == 1:
                    # Top-1 selection (most efficient, mentioned in Section 3.4)
                    # The paper shows this is the most efficient approach in Table 1
                    expert_idx = router_probs.argmax().item()
                    selected_expert = experts[expert_idx]
                    
                    # Log which expert is being used (every 10 steps to avoid spam)
                    if i % 10 == 0:
                        logger.debug(f"Step {i}: Using expert {expert_indices[selected_expert]} with probability {router_probs[0, expert_idx]:.4f}")
                    
                    pred_flow = selected_expert(latent, t_tensor, text_embeddings)
                else:
                    # Full ensemble (Equation 7 in the paper)
                    # u_t(x_t) = sum_k p_k(x_t, t) * u_t^k(x_t)
                    expert_flows = []
                    
                    # Log expert probabilities (every 10 steps to avoid spam)
                    if i % 10 == 0:
                        top_experts = torch.topk(router_probs[0], min(3, len(experts)))
                        logger.debug(f"Step {i}: Top experts: " + ", ".join([
                            f"Expert {idx.item()} ({prob.item():.4f})" 
                            for idx, prob in zip(top_experts.indices, top_experts.values)
                        ]))
                    
                    for expert in experts:
                        expert_flow = expert(latent, t_tensor, text_embeddings)
                        expert_flows.append(expert_flow)
                    
                    # Combine expert flows according to router probabilities
                    pred_flow = 0
                    for k in range(len(experts)):
                        pred_flow += router_probs[0, k] * expert_flows[k]
                
                # Update latent with flow matching (Equation 8 in the paper)
                # x_{t+dt} = x_t + sigma_t * u_t(x_t) * dt
                # This is the Euler integration step for the flow ODE
                dt = ts[i+1] - t
                latent = latent + sigma_t * pred_flow * dt
        
        # Decode final latent (Section 3.4)
        # The paper mentions dividing by alpha_t to account for the scaling
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return vae_wrapper.decode(latent / alphas[-1])

    def save_checkpoints(self):
        if self.rank == 0:
            os.makedirs(self.config.save_dir, exist_ok=True)
            
            # Save expert checkpoints using their save_checkpoint method
            for trainer in self.expert_trainers:
                checkpoint_path = trainer.save_checkpoint(self.config.save_dir, self.current_step)
                logger.info(f"Saved expert {trainer.expert_idx} checkpoint to {checkpoint_path}")
            
            # Save router
            router_path = f"{self.config.save_dir}/router_step{self.current_step}.pt"
            torch.save(
                self.router_trainer.router.state_dict(),
                router_path
            )
            logger.info(f"Saved router checkpoint to {router_path}")

    def perform_distillation(self):
        """
        Distill the ensemble into a single model as described in Section 3.6 of the paper
        
        The paper uses a teacher-student training procedure where the student model
        is trained to match the predictions of the teacher ensemble.
        """
        logger.info("Starting distillation process as described in Section 3.6...")
        
        # Initialize student model with same architecture as experts
        student = ExpertDiT(self.config).to(self.device)
        
        # Paper-recommended optimizer settings for distillation from config
        optimizer = AdamW8bit(
            student.parameters(), 
            lr=self.config.distill_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Create distillation dataset (subset of training data)
        distill_dataset = Subset(
            self.full_dataset, 
            indices=range(min(self.config.distill_samples, len(self.full_dataset)))
        )
        
        # Create dataloader for distillation using config parameters
        distill_loader = DataLoader(
            distill_dataset, 
            batch_size=self.config.distill_batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Create a mapping from cluster ID to expert trainer for efficient lookup
        expert_map = {trainer.expert_idx: trainer for trainer in self.expert_trainers}
        logger.info(f"Created expert map with {len(expert_map)} experts")
        
        # Distillation training loop
        for epoch in range(self.config.distill_epochs):
            total_loss = 0
            
            for batch in tqdm(distill_loader, desc=f"Distillation Epoch {epoch}"):
                images = batch["image"].to(self.device)
                
                # Use mixed precision from config for consistency
                scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_mixed_precision)
                
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    # VAE encoding
                    vae = VAEWrapper(self.device, self.config)
                    latents = vae.encode(images)
                    
                    # Random timestep and noise (same as in expert training)
                    t = torch.rand(latents.size(0), device=self.device)
                    noise = torch.randn_like(latents)
                    
                    # Forward process (same as in expert training)
                    alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
                    sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
                    xt = alpha_t * latents + sigma_t * noise
                    
                    # Get teacher ensemble prediction
                    # Following Section 3.6, we use the cluster label to select the teacher expert
                    with torch.no_grad():
                        # Get cluster assignments for each sample
                        clusters = batch["cluster"].to(self.device)
                        
                        # Get predictions from selected experts based on cluster assignments
                        teacher_flow = torch.zeros_like(xt)
                        for b in range(xt.size(0)):
                            cluster_id = clusters[b].item()
                            # Use the expert_map to find the correct expert trainer
                            if cluster_id in expert_map:
                                expert_trainer = expert_map[cluster_id]
                                teacher_flow[b:b+1] = expert_trainer.expert(
                                    xt[b:b+1], 
                                    t[b:b+1]
                                )
                            else:
                                # Fallback if cluster ID doesn't match any expert
                                logger.warning(f"Cluster ID {cluster_id} not found in expert map, using expert 0")
                                teacher_flow[b:b+1] = self.expert_trainers[0].expert(
                                    xt[b:b+1], 
                                    t[b:b+1]
                                )
                    
                    # Student prediction
                    student_flow = student(xt, t)
                    
                    # Distillation loss (Equation from Section 3.6)
                    # Ldistill(θ) = E_{t,x_0,ε}[||vθ,t(xt) - vteacher,t(xt)||²]
                    loss = torch.nn.functional.mse_loss(student_flow, teacher_flow)
                
                # Optimization with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
            
            avg_loss = total_loss/len(distill_loader)
            logger.info(f"Distillation Epoch {epoch} Loss: {avg_loss:.4f}")
            
            # Log distillation metrics to wandb using helper method
            log_data = {
                "distillation/loss": avg_loss,
                "distillation/epoch": epoch,
                "step": self.current_step
            }
            self._log_to_wandb_async(log_data)
        
        # Save distilled model
        distilled_path = f"{self.config.save_dir}/distilled_model.pt"
        torch.save(student.state_dict(), distilled_path)
        logger.info(f"Distilled model saved to {distilled_path}")

    def create_progress_bar(self, expert_idx):
        return tqdm(total=self.config.save_interval, 
                  desc=f"Expert {expert_idx}") if self.rank == 0 else None

    def log_metrics(self, expert_idx, step, loss):
        if step % self.config.log_every_n_steps == 0:
            logger.info(f"Expert {expert_idx} | Step {step} | Loss: {loss:.4f}")
            
            # Log metrics to wandb using helper method
            log_data = {
                f"expert_{expert_idx}/train_loss": loss,
                "step": self.current_step + step
            }
            self._log_to_wandb_async(log_data)

    def log_training_metrics(self):
        """
        Log training metrics to wandb.
        This includes learning rates, memory usage, and training speed.
        Uses non-blocking logging to avoid performance impact.
        """
        if self.rank == 0 and wandb.run is not None:
            # Get current learning rates
            expert_lr = self.expert_trainers[0].optimizer.param_groups[0]['lr']
            router_lr = self.router_trainer.optimizer.param_groups[0]['lr']
            
            # Get GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)    # GB
            else:
                memory_allocated = 0
                memory_reserved = 0
            
            # Calculate training speed (steps per second)
            current_time = time.time()
            if hasattr(self, 'last_log_time'):
                steps_since_last_log = self.current_step - self.last_log_step
                time_since_last_log = current_time - self.last_log_time
                steps_per_second = steps_since_last_log / time_since_last_log if time_since_last_log > 0 else 0
            else:
                steps_per_second = 0
            
            self.last_log_time = current_time
            self.last_log_step = self.current_step
            
            # Prepare metrics data and log using helper method
            log_data = {
                "training/expert_lr": expert_lr,
                "training/router_lr": router_lr,
                "system/memory_allocated_gb": memory_allocated,
                "system/memory_reserved_gb": memory_reserved,
                "system/steps_per_second": steps_per_second,
                "step": self.current_step
            }
            self._log_to_wandb_async(log_data)

    def log_lr_schedules(self):
        """
        Log the learning rate schedules to wandb.
        This is useful for visualizing the learning rate decay over time.
        Uses non-blocking logging to avoid performance impact.
        
        This aligns with Section 4.1 of the paper, which discusses the training
        details including learning rate schedules. The paper mentions using
        cosine annealing for learning rate decay, which is visualized by this method.
        """
        if self.rank == 0 and wandb.run is not None:
            # Create arrays for steps and learning rates
            steps = list(range(0, self.config.num_steps, 100))
            expert_lrs = []
            router_lrs = []
            
            # Get the learning rate schedulers
            expert_scheduler = self.expert_trainers[0].scheduler
            router_scheduler = self.router_trainer.scheduler
            
            # Get the initial learning rates
            expert_lr = self.config.learning_rate
            router_lr = self.config.router_learning_rate
            
            # Calculate learning rates at each step
            for step in steps:
                # Calculate expert learning rate
                if expert_scheduler is not None:
                    for _ in range(step):
                        expert_scheduler.step()
                    expert_lr = expert_scheduler.get_last_lr()[0]
                    # Reset scheduler
                    expert_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.expert_trainers[0].optimizer, 
                        T_max=self.config.num_steps
                    )
                
                # Calculate router learning rate
                if router_scheduler is not None:
                    for _ in range(step):
                        router_scheduler.step()
                    router_lr = router_scheduler.get_last_lr()[0]
                    # Reset scheduler
                    router_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.router_trainer.optimizer, 
                        T_max=self.config.num_steps
                    )
                
                expert_lrs.append(expert_lr)
                router_lrs.append(router_lr)
            
            # Create a table for the learning rate schedules
            lr_table = wandb.Table(
                columns=["step", "expert_lr", "router_lr"],
                data=[[step, expert_lr, router_lr] for step, expert_lr, router_lr in zip(steps, expert_lrs, router_lrs)]
            )
            
            # Log the table using helper method
            self._log_to_wandb_async({"lr_schedules": lr_table})

    def log_model_graph(self):
        """
        Log the model architecture as a graph to wandb.
        This is useful for visualizing the model structure.
        Uses non-blocking logging to avoid performance impact.
        """
        if self.rank == 0 and wandb.run is not None:
            try:
                # Use a separate thread for model graph logging to avoid blocking
                def log_model_graphs():
                    # Log router model graph with non-blocking logging
                    wandb.watch(
                        self.router_trainer.router, 
                        log="all", 
                        log_freq=100,
                        log_graph=True
                    )
                    
                    # Log expert model graphs (just the first one to avoid clutter)
                    if len(self.expert_trainers) > 0:
                        wandb.watch(
                            self.expert_trainers[0].expert, 
                            log="all", 
                            log_freq=100,
                            log_graph=True
                        )
                
                # Start the logging in a separate thread
                threading.Thread(
                    target=log_model_graphs,
                    daemon=True
                ).start()
                
                logger.info("Model graphs logged to wandb (non-blocking)")
            except Exception as e:
                logger.warning(f"Failed to log model graph to wandb: {e}")

    def log_dataset_stats(self):
        """
        Log dataset statistics to wandb.
        This includes the number of samples per expert and the distribution of samples.
        Uses non-blocking logging to avoid performance impact.
        
        This aligns with Section 3.2 of the paper, which discusses the initial
        clustering and data distribution among experts. Monitoring the sample
        distribution helps ensure that experts receive a balanced workload,
        which is important for efficient training.
        """
        if self.rank == 0 and wandb.run is not None:
            try:
                # Get the number of samples per expert
                expert_sample_counts = []
                for expert_idx, expert_trainer in enumerate(self.expert_trainers):
                    if hasattr(expert_trainer, 'dataloader') and expert_trainer.dataloader is not None:
                        num_samples = len(expert_trainer.dataloader.dataset)
                        expert_sample_counts.append((expert_idx, num_samples))
                
                # Create a bar chart of the sample distribution
                if expert_sample_counts:
                    expert_indices, sample_counts = zip(*expert_sample_counts)
                    
                    # Create a wandb bar chart
                    data = [[f"Expert {idx}", count] for idx, count in zip(expert_indices, sample_counts)]
                    table = wandb.Table(data=data, columns=["Expert", "Sample Count"])
                    
                    # Log the chart using helper method
                    log_data = {
                        "dataset/sample_distribution": wandb.plot.bar(
                            table, "Expert", "Sample Count", 
                            title="Sample Distribution Across Experts"
                        ),
                        "dataset/total_samples": sum(sample_counts),
                        "dataset/num_experts_with_data": len(expert_sample_counts)
                    }
                    
                    self._log_to_wandb_async(log_data)
                    
                    logger.info(f"Dataset statistics logged to wandb: {len(expert_sample_counts)} experts with data")
            except Exception as e:
                logger.warning(f"Failed to log dataset statistics to wandb: {e}")

    def log_cluster_assignments(self, clusters):
        """
        Log the cluster assignments to wandb.
        This includes a confusion matrix of how samples moved between clusters.
        Uses non-blocking logging to avoid performance impact.
        
        This aligns with Section 3.3 of the paper, which discusses the dynamic
        reclustering process. The confusion matrix helps visualize how data points
        move between experts during reclustering, which is a key aspect of the
        decentralized training approach.
        
        Args:
            clusters: The new cluster assignments
        """
        if self.rank == 0 and wandb.run is not None and hasattr(self, 'previous_clusters'):
            try:
                # Create a confusion matrix of how samples moved between clusters
                confusion_matrix = np.zeros((self.config.num_experts, self.config.num_experts), dtype=np.int32)
                
                # Count how many samples moved from each previous cluster to each new cluster
                for prev_cluster, new_cluster in zip(self.previous_clusters, clusters):
                    if prev_cluster < self.config.num_experts and new_cluster < self.config.num_experts:
                        confusion_matrix[prev_cluster, new_cluster] += 1
                
                # Create a wandb confusion matrix
                labels = [f"Expert {i}" for i in range(self.config.num_experts)]
                
                # Prepare data for logging
                log_data = {
                    "clustering/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=np.zeros(0),  # Not used when providing the matrix directly
                        preds=np.zeros(0),   # Not used when providing the matrix directly
                        class_names=labels,
                        matrix=confusion_matrix,
                        title="Cluster Reassignments"
                    ),
                    "clustering/step": self.current_step,
                    "clustering/num_reassignments": np.sum(confusion_matrix) - np.trace(confusion_matrix)
                }
                
                # Log using helper method
                self._log_to_wandb_async(log_data)
                
                logger.info(f"Cluster assignments logged to wandb: {np.sum(confusion_matrix)} samples, {np.sum(confusion_matrix) - np.trace(confusion_matrix)} reassignments")
            except Exception as e:
                logger.warning(f"Failed to log cluster assignments to wandb: {e}")
        
        # Store the current clusters for the next reclustering
        self.previous_clusters = clusters.copy()

    def log_expert_specialization(self):
        """
        Log expert specialization metrics to wandb.
        This analyzes how specialized each expert has become.
        Uses non-blocking logging to avoid performance impact.
        
        This aligns with Section 3.5 of the paper, which discusses the specialization
        of experts during training. The entropy and max probability metrics help
        quantify how decisively the router assigns samples to experts.
        """
        if self.rank == 0 and wandb.run is not None:
            try:
                # Create a dataset with a small subset of samples
                subset_size = min(1000, len(self.full_dataset))
                subset_indices = np.random.choice(len(self.full_dataset), subset_size, replace=False)
                subset_dataset = Subset(self.full_dataset, subset_indices)
                subset_loader = DataLoader(subset_dataset, batch_size=self.config.batch_size, shuffle=False)
                
                # Get router predictions for each sample
                router_probs_list = []
                with torch.no_grad():
                    self.router_trainer.router.eval()
                    
                    for batch in subset_loader:
                        images = batch['image'].to(self.device)
                        # Sample random timesteps
                        t = torch.rand(images.shape[0], device=self.device)
                        
                        # Get latents
                        vae_wrapper = VAEWrapper(self.device, self.config)
                        latents = vae_wrapper.encode(images)
                        
                        # Get router probabilities
                        router_logits = self.router_trainer.router(latents, t)
                        router_probs = F.softmax(router_logits, dim=-1)
                        router_probs_list.append(router_probs.cpu().numpy())
                
                # Concatenate all router probabilities
                all_router_probs = np.concatenate(router_probs_list, axis=0)
                
                # Calculate specialization metrics
                entropy = -np.sum(all_router_probs * np.log(all_router_probs + 1e-10), axis=1)
                max_probs = np.max(all_router_probs, axis=1)
                avg_entropy = np.mean(entropy)
                avg_max_prob = np.mean(max_probs)
                
                # Calculate expert utilization
                expert_utilization = np.mean(all_router_probs, axis=0)
                
                # Prepare metrics data
                metrics_data = {
                    "specialization/avg_entropy": avg_entropy,
                    "specialization/avg_max_prob": avg_max_prob,
                    "step": self.current_step
                }
                
                # Log metrics using helper method
                self._log_to_wandb_async(metrics_data)
                
                # Create expert utilization chart
                data = [[f"Expert {i}", util] for i, util in enumerate(expert_utilization)]
                table = wandb.Table(data=data, columns=["Expert", "Utilization"])
                
                # Log chart using helper method
                chart_data = {
                    "specialization/expert_utilization": wandb.plot.bar(
                        table, "Expert", "Utilization", 
                        title="Expert Utilization"
                    ),
                    "step": self.current_step
                }
                
                self._log_to_wandb_async(chart_data)
                
                logger.info(f"Expert specialization metrics logged to wandb: avg_entropy={avg_entropy:.4f}, avg_max_prob={avg_max_prob:.4f}")
            except Exception as e:
                logger.warning(f"Failed to log expert specialization metrics to wandb: {e}")

    def _log_to_wandb_async(self, data):
        """
        Helper method to log data to wandb asynchronously.
        This avoids blocking the main training loop.
        
        Args:
            data: Dictionary of data to log to wandb
        """
        if self.rank == 0 and wandb.run is not None:
            threading.Thread(
                target=wandb.log,
                args=(data,),
                daemon=True
            ).start() 