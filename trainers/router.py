"""Router trainer for Decentralized Diffusion Models."""

import torch
import torch.nn as nn
import math
from bitsandbytes.optim import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, CPUOffload
from torch.nn import functional as F
from tqdm import tqdm

from models.router import RouterModel
from utils.checkpoint import save_model_checkpoint, load_model_checkpoint

class RouterTrainer:
    """Trainer for the router model in DDM"""
    def __init__(self, config, device, rank, world_size=None):
        # Initialize parameters
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size or 1
        
        # Create base router model
        base_router = RouterModel(config).to(device)
        
        # Apply FSDP if world_size > 1
        if self.world_size > 1:
            # Configure FSDP settings based on config
            # Sharding strategy
            if config.fsdp_sharding_strategy == "FULL_SHARD":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            elif config.fsdp_sharding_strategy == "SHARD_GRAD_OP":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            else:
                sharding_strategy = ShardingStrategy.FULL_SHARD
                
            # CPU offload
            cpu_offload = CPUOffload(offload_params=config.fsdp_cpu_offload)
            
            # Backward prefetch
            if config.fsdp_backward_prefetch == "BACKWARD_PRE":
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
            elif config.fsdp_backward_prefetch == "BACKWARD_POST":
                backward_prefetch = BackwardPrefetch.BACKWARD_POST
            else:
                backward_prefetch = BackwardPrefetch.BACKWARD_PRE
                
            # Auto wrap policy
            if config.fsdp_auto_wrap_policy == "DEFAULT":
                auto_wrap_policy = default_auto_wrap_policy
            elif config.fsdp_auto_wrap_policy == "SIZE_BASED":
                auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=config.fsdp_min_num_params)
            else:
                auto_wrap_policy = default_auto_wrap_policy
            
            # Apply FSDP to shard model across all GPUs
            self.router = FSDP(
                base_router,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                backward_prefetch=backward_prefetch,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=True  # Allow easier parameter access
            )
            
            if rank == 0:
                print(f"Initialized SHARDED Router across {self.world_size} GPUs")
        else:
            # Just use the base model without FSDP
            self.router = base_router
            
        # Paper-recommended optimizer settings
        self.optimizer = AdamW8bit(
            self.router.parameters(),
            lr=config.router_learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Paper-recommended learning schedule
        # Add warmup steps if not in config
        self.warmup_steps = getattr(config, 'warmup_steps', int(0.05 * config.num_steps))
        self.total_steps = getattr(config, 'num_steps', 400000)
        
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(step/self.warmup_steps, 1.0) 
            if step < self.warmup_steps 
            else 0.5*(1 + math.cos(math.pi*(step - self.warmup_steps)/self.total_steps))
        )

    def train_epoch(self, loader):
        """
        Implements Algorithm 1 from paper (router training)
        
        This trains the router model to predict which expert should handle
        each sample, as described in Section 3.3 of the paper.
        """
        # Add gradient isolation
        for param in self.router.parameters():
            param.requires_grad_(False)  # Freeze first
        
        # Only unfreeze router-specific params
        for name, param in self.router.named_parameters():
            if "classifier" in name or "cls_token" in name:
                param.requires_grad_(True)
        
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            # Get images and cluster assignments (Section 3.3)
            # The cluster assignment k* is the ground truth for router training
            images = batch["image"].to(self.device)
            clusters = batch["cluster"].to(self.device)  # k* in Algorithm 1
            
            # Sample random timesteps t ∈ [0, 1] (Section 3.3)
            # The paper uses uniform sampling of t in [0, 1]
            t = torch.rand(images.size(0), device=self.device)
            
            # Sample random noise (Section 3.3)
            # ε ~ N(0, I) as in Algorithm 1
            noise = torch.randn_like(images)
            
            # Forward process using cosine schedule (Section 3.3)
            # x_t = alpha_t * x_0 + sigma_t * noise
            # This follows the cosine schedule in the paper
            alpha_t = torch.cos(t * math.pi/2)[:,None,None,None]
            sigma_t = torch.sin(t * math.pi/2)[:,None,None,None]
            xt = alpha_t * images + sigma_t * noise
            
            # Router prediction (Equation 5 in the paper)
            # The router predicts which expert should handle this sample
            # z = rθ(xt, t) ∈ R^K where K is the number of experts
            logits = self.router(xt, t)
            
            # Cross-entropy loss for router (Section 3.3)
            # L_router = E_{x_0,t}[-log p_k*(x_t, t)]
            # where k* is the cluster assignment for x_0
            # This is implemented as cross-entropy between logits and cluster labels
            loss = self.criterion(logits, clusters)
            
            # Add confidence thresholding
            if hasattr(self.config, 'router_confidence_threshold') and self.config.router_confidence_threshold > 0:
                probs = torch.softmax(logits, dim=1)
                max_prob = probs.max(dim=1)[0]
                mask = (max_prob > self.config.router_confidence_threshold).float()
                loss = (loss * mask).mean()
            
            # Optimization (Section 4.1)
            # The paper uses AdamW with weight decay
            self.optimizer.zero_grad()
            loss.backward()
            # Paper-recommended gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.router.parameters(), 
                max_norm=self.config.max_grad_norm,  # Should be 1.0 in config
                norm_type=2.0
            )
            self.optimizer.step()
            self.lr_scheduler.step()  # Update learning rate
            
            total_loss += loss.item()
            num_batches += 1
        
        # Return average loss over the epoch
        return total_loss / num_batches

    def calibrate_confidence(self, val_loader):
        """
        Enhanced temperature scaling for better router calibration
        
        Paper-recommended temperature scaling with improvements:
        - Early stopping based on validation accuracy
        - Multiple random restarts for better optimization
        - Target-aware temperature adjustment
        """
        # Freeze all parameters except temperature
        for param in self.router.parameters():
            param.requires_grad = False
        
        # Reset temperature parameter
        self.router.temperature = nn.Parameter(torch.ones(1, device=self.device))
        self.router.temperature.requires_grad = True
        
        # Create optimizer with conservative learning rate
        optimizer = torch.optim.LBFGS([self.router.temperature], lr=0.01, max_iter=20)
        
        # Collect validation data for calibration
        val_inputs = []
        val_targets = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Collecting validation data"):
                images = batch["image"].to(self.device)
                clusters = batch["cluster"].to(self.device)
                
                # Sample multiple timesteps for robust calibration
                for _ in range(min(3, len(images))):  # Sample up to 3 timesteps
                    t = torch.rand(images.size(0), device=self.device)
                    noise = torch.randn_like(images)
                    xt = torch.cos(t * math.pi/2)[:,None,None,None] * images + \
                         torch.sin(t * math.pi/2)[:,None,None,None] * noise
                    
                    val_inputs.append((xt, t))
                    val_targets.append(clusters)
        
        # Define model evaluation function for LBFGS
        def eval_fn():
            optimizer.zero_grad()
            total_loss = 0
            total_samples = 0
            
            # Calculate loss on all collected samples
            for (xt, t), clusters in zip(val_inputs, val_targets):
                # Forward pass with current temperature
                logits = self.router(xt, t)
                loss = F.cross_entropy(logits, clusters)
                
                # Weight loss by batch size
                batch_size = xt.size(0)
                total_loss += loss * batch_size
                total_samples += batch_size
            
            # Calculate average loss
            avg_loss = total_loss / total_samples
            
            # Backward pass
            avg_loss.backward()
            
            # Return loss value
            return avg_loss
        
        # Track best result
        best_loss = float('inf')
        best_temp = 1.0
        
        # Try multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            # Initialize temperature randomly
            with torch.no_grad():
                init_temp = 0.5 + 1.5 * torch.rand(1, device=self.device)
                self.router.temperature.copy_(init_temp)
            
            # Run L-BFGS optimization with early stopping
            prev_loss = float('inf')
            for iteration in range(5):  # Outer iterations
                current_loss = optimizer.step(eval_fn).item()
                
                # Early stopping
                if abs(current_loss - prev_loss) < 1e-4:
                    break
                    
                prev_loss = current_loss
                
            # Check if this restart gave better results
            final_loss = prev_loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_temp = self.router.temperature.item()
                
        # Set to best temperature
        with torch.no_grad():
            self.router.temperature.copy_(torch.tensor([best_temp], device=self.device))
            
        # Calculate calibration metrics
        with torch.no_grad():
            correct = 0
            total = 0
            confidence_sum = 0
            
            for (xt, t), clusters in zip(val_inputs, val_targets):
                # Get predictions
                logits = self.router(xt, t)
                probs = F.softmax(logits, dim=1)
                
                # Calculate accuracy
                _, preds = logits.max(1)
                correct += (preds == clusters).sum().item()
                total += clusters.size(0)
                
                # Calculate average confidence
                confidence = probs.max(1)[0].mean().item()
                confidence_sum += confidence * clusters.size(0)
                
            # Print metrics
            accuracy = correct / total
            avg_confidence = confidence_sum / total
            
            if self.rank == 0:
                print(f"Router calibration results:")
                print(f"  Temperature: {best_temp:.4f}")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Avg confidence: {avg_confidence:.4f}")
                print(f"  Gap (confidence - accuracy): {avg_confidence - accuracy:.4f}")
        
        # Reset gradients for training
        for param in self.router.parameters():
            param.requires_grad = True
            
        return best_temp

    def save_checkpoint(self, save_dir, step):
        """Save router checkpoint using centralized utility"""
        # Create checkpoint path
        checkpoint_path = f"{save_dir}/router_step{step}.pt"
        
        # Create metadata
        metadata = {
            'step': step,
            'temperature': self.router.temperature.item(),
            'config': {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        }
        
        # Save using the centralized utility
        return save_model_checkpoint(
            model=self.router,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            metadata=metadata,
            is_fsdp=isinstance(self.router, FSDP)
        )
        
    def load_checkpoint(self, checkpoint_path):
        """Load router checkpoint using centralized utility"""
        # Load using the centralized utility
        metadata = load_model_checkpoint(
            model=self.router,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            path=checkpoint_path,
            is_fsdp=isinstance(self.router, FSDP),
            device=self.device
        )
        
        return metadata

    # Add a new method to improve router predictions during inference
    def get_routing_weights(self, xt, t, top_k=None, temperature=None):
        """
        Get routing weights with improved temperature scaling
        
        Args:
            xt: Input latent [B, C, H, W]
            t: Timesteps [B]
            top_k: Number of experts to select (None=all)
            temperature: Override temperature (None=use calibrated value)
            
        Returns:
            weights: Routing weights [B, num_experts] 
            indices: Selected expert indices [B, top_k]
        """
        with torch.no_grad():
            # Get router predictions
            logits = self.router(xt, t)
            
            # Apply temperature scaling for calibrated predictions
            if temperature is not None:
                # Use provided temperature
                scaled_logits = logits / temperature
            else:
                # Use the router's calibrated temperature
                scaled_logits = logits / self.router.temperature
            
            # Apply softmax for probabilities
            probs = F.softmax(scaled_logits, dim=1)
            
            if top_k is not None and top_k < probs.size(1):
                # Get top-k expert indices and probabilities
                weights, indices = probs.topk(min(top_k, probs.size(1)), dim=1)
                
                # Normalize weights to sum to 1
                weights = weights / weights.sum(dim=1, keepdim=True)
            else:
                # Use all experts
                weights = probs
                indices = torch.arange(probs.size(1), device=probs.device).expand(probs.size(0), -1)
            
            return weights, indices

    def forward(self, x_t, timesteps, clip_embeddings=None, temperature=1.0):
        """
        Forward pass for router model
        
        Args:
            x_t: Noisy data at timestep t
            timesteps: Timestep values
            clip_embeddings: Optional text embeddings for conditional generation
            temperature: Temperature for softmax (higher = more uniform)
            
        Returns:
            expert_weights: Expert selection weights [batch_size, num_experts]
        """
        # Get router outputs (logits)
        router_logits = self.router(x_t, timesteps, clip_embeddings)
        
        # Apply temperature scaling and convert to probabilities
        expert_weights = F.softmax(router_logits / temperature, dim=-1)
        
        return expert_weights
    
    def predict_expert(self, x_t, timesteps, clip_embeddings=None, temperature=1.0, top_k=1, use_threshold=False, threshold=0.1):
        """
        Predict which expert(s) to use for a given input
        
        Args:
            x_t: Noisy data at timestep t
            timesteps: Timestep values
            clip_embeddings: Optional text embeddings
            temperature: Temperature for softmax
            top_k: Number of top experts to select
            use_threshold: Whether to use probability threshold
            threshold: Minimum probability threshold
            
        Returns:
            selected_experts: List of selected expert indices for each batch item
            expert_weights: Expert weights for each batch item
        """
        batch_size = x_t.shape[0]
        
        # Get expert weights
        expert_weights = self.forward(x_t, timesteps, clip_embeddings, temperature)
        
        if use_threshold:
            # Select experts with probability above threshold
            selected_experts = [
                torch.where(weights >= threshold)[0].cpu().tolist()
                for weights in expert_weights
            ]
            
            # Ensure at least one expert is selected
            for i in range(batch_size):
                if not selected_experts[i]:
                    # If no experts selected, use top expert
                    selected_experts[i] = [torch.argmax(expert_weights[i]).item()]
        else:
            # Select top-k experts
            _, indices = torch.topk(expert_weights, k=min(top_k, expert_weights.size(1)), dim=1)
            selected_experts = [indices[i].cpu().tolist() for i in range(batch_size)]
            
        return selected_experts, expert_weights
            
    def train_step(self, batch):
        """
        Train router for one step following Algorithm 1 from the paper
        
        Args:
            batch: Dict with 'image', 'cluster', and optional 'text_embedding'
            
        Returns:
            Loss value
        """
        # Move batch to device
        images = batch["image"].to(self.device)
        cluster_labels = batch["cluster"].to(self.device)
        text_embeddings = batch.get("text_embedding", None)
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.device)
            
        # Create diffusion timesteps
        batch_size = images.shape[0]
        
        # Sample random timestep for each sample (uniform in [0, 1])
        t = torch.rand(batch_size, device=self.device)
        
        # Convert to timestep indices (t ∈ [0, 1000))
        timesteps = (t * 1000).long()
        
        # Create noise
        noise = torch.randn_like(images)
        
        # Forward diffusion
        alpha_t = torch.cos(t.view(-1, 1, 1, 1) * torch.pi/2)
        sigma_t = torch.sin(t.view(-1, 1, 1, 1) * torch.pi/2)
        noisy_images = alpha_t * images + sigma_t * noise
        
        # Paper Algorithm 1: Train router to predict cluster for each noisy image
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            # Forward pass
            logits = self.router(noisy_images, timesteps, text_embeddings)
            
            # Compute cross entropy loss
            loss = self.criterion(logits, cluster_labels)
            
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.router.parameters(), self.config.max_grad_norm)
            
        self.optimizer.step()
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return loss.item()
