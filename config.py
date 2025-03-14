"""Configuration for Decentralized Diffusion Models."""

import os

class DDMConfig:
    """Configuration for Decentralized Diffusion Models"""
    def __init__(self):
        # Model architecture
        self.hidden_dim = 1152  # DiT-XL dimension
        self.num_layers = 28     # DiT-XL depth
        self.num_heads = 16      # DiT-XL heads
        self.ffn_dim = 3072      # DiT-XL MLP ratio
        self.batch_size = 1      # Reduced from 16 to 1 to address OOM issues
        
        # DDM specific settings
        self.num_experts = 8     # Number of expert models (paper recommends 8)
        self.router_hidden_dim = 512  # Smaller router dimension
        
        # Expert Memory Management (new section)
        self.expert_swap_strategy = "LRU"       # LRU, FIFO, or RANDOM
        self.max_experts_in_memory = 3          # Maximum experts to keep in GPU memory
        self.expert_offload_to_cpu = True       # Whether to offload experts to CPU
        self.expert_prefetch_next = True        # Prefetch next predicted expert
        self.sample_expert_cache_size = 2       # Number of experts to cache during sampling
        
        # Training settings
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.adam_betas = (0.9, 0.99)
        
        # Memory optimization settings
        self.use_mixed_precision = True  # Use mixed precision training (fp16)
        self.gradient_accumulation_steps = 1  # Accumulate gradients over multiple steps
        self.use_gradient_checkpointing = True  # Use gradient checkpointing to save memory
        
        # FSDP (Fully Sharded Data Parallel) settings
        self.fsdp_sharding_strategy = "FULL_SHARD"  # Fully shard parameters, gradients, and optimizer states
        self.fsdp_cpu_offload = True
        self.fsdp_activation_checkpointing = True
        self.fsdp_sync_module_states = True
        self.fsdp_use_orig_params = True
        self.fsdp_limit_all_gathers = True
        self.fsdp_forward_prefetch = True
        self.fsdp_min_num_params = 1e6  # Minimum number of parameters for a layer to be wrapped (1M)
        self.fsdp_auto_wrap_policy = "DEFAULT"  # Default auto wrap policy
        self.fsdp_backward_prefetch = "BACKWARD_PRE"  # Prefetch parameters before backward pass

        # Data settings
        self.patch_size = 32
        self.image_size = 512
        self.dataset_path = "/home/alex/workspace/datasets/danbooru2025"  # Path to training dataset
        self.dataset_size = self._calculate_dataset_size()  # Add this line
        self.num_workers = 2     # DataLoader workers
        self.pin_memory = True   # Pin memory for faster data transfer
        
        # Enhanced Clustering Settings (new section)
        self.clustering_method = "two_stage"    # "two_stage" (recommended), "direct", or "kmeans"
        self.fine_clusters = 1024               # Number of fine-grained clusters (paper recommendation)
        self.use_hierarchical_clustering = True # Use hierarchical consolidation
        self.feature_sub_batch_size = 32        # Sub-batch size for feature extraction
        self.feature_extraction_model = "dinov2" # "dinov2", "clip", or "custom"
        self.use_feature_cache = True           # Cache extracted features
        self.cluster_cache_path = "cache"       # Path to store feature and cluster caches
        
        # Training settings
        self.num_steps = 400000  
        self.log_dir = "runs/ddm"
        self.save_dir = "checkpoints/ddm"
        self.checkpoint_dir = "checkpoints/ddm"  # Added to match save_dir
        self.sample_dir = "samples/ddm"  # Directory to save generated samples
        self.expert_steps_per_router = 1000  # Number of expert steps between router training
        
        # Improved Router Settings (new section) 
        self.router_architecture = "efficient"  # "efficient" or "standard"
        self.router_attention_layers = 2        # Number of attention layers in router
        self.router_calibration_method = "temperature" # "temperature" or "platt"
        self.router_regularization = 0.01       # Regularization for router training
        self.router_confidence_threshold = 0.7  # Confidence threshold for expert selection
        
        # Inference settings
        self.use_top_k = 1  # Number of experts to use at inference time
        self.inference_steps = 50  # Number of steps for sampling
        self.inference_prompt = "1girl"  # Default prompt
        self.use_nucleus_sampling = False  # Whether to use nucleus sampling
        self.nucleus_threshold = 0.9  # Threshold for nucleus sampling
        self.inference_temperature = 1.0  # Temperature for expert probabilities
        
        # VAE settings
        self.vae_model = "AuraDiffusion/16ch-vae"
        self.latent_channels = 16
        
        # CLIP settings
        self.clip_model = "openai/clip-vit-large-patch14"
        
        # Flow matching parameters
        self.sigma = 0.8  # Flow matching noise scale
        self.cfg_scale = 7.5  # Classifier-free guidance scale
        self.loss_type = 'huber'  # Loss type for flow matching
        self.nucleus_threshold = 0.9  # Added for nucleus sampling
        self.do_distillation = True  # Added for distillation
        
        # Logging and validation
        self.log_every_n_steps = 1  # Log every n steps
        self.validation_interval = 1000  # Steps between validation
        self.save_interval = 5000  # Steps between saving checkpoints
        self.recluster_interval = 50000  # Steps between reclustering
        self.distill_interval = 100000  # Steps between distillation
        
        # Paper-recommended hyperparameters
        self.router_batch_size = 1  # Batch size for router training
        self.expert_batch_size = 1  # Per-expert batch size
        self.router_learning_rate = 3e-5   # Different from expert LR
        self.recluster_epochs = 3          # Paper recommends 3 passes
        self.ema_decay = 0.9999            # For model stability
        self.max_grad_norm = 1.0           # Gradient clipping
        self.fid_sample_size = 50000       # For meaningful FID
        self.inception_metrics_interval = 10000  # Steps between eval
        self.diversity_threshold = 0.75    # Novelty measurement
        
        # Feature extraction settings
        self.feature_batch_size = 1
        self.feature_workers = 4
        self.dino_size = 518  # Size for DINO feature extraction
        
        # Enhanced Distillation Settings (new section)
        self.distill_lr = 1e-5
        self.distill_batch_size = 1
        self.distill_epochs = 10
        self.distill_samples = 10000  # Number of samples for distillation
        self.distill_balance_clusters = True  # Whether to balance samples across clusters
        self.distill_ema_decay = 0.9999  # EMA decay for distilled model
        self.distill_loss_type = "mse"  # "mse", "huber", or "l1"
        self.distill_warmup_ratio = 0.1  # Warmup ratio for distillation
        
        # Other configurations
        self.max_loaded_experts = 3  # Paper recommends 2-3 for 8 experts
        self.validation_topk = 1     # Paper Section 4.3 recommendation
        self.validation_samples = 4  # Default from code 
        
        # Add calibration parameters
        self.calibration_interval = 1000  # Calibrate every 1000 steps
        self.router_confidence_threshold = 0.7  # 0.7 recommended by paper
        self.fallback_expert_idx = 0  # Default fallback expert

    def _calculate_dataset_size(self):
        """Calculate dataset size from directory (only called on rank 0)"""
        if not os.path.exists(self.dataset_path):
            return 0
            
        return len([
            f for f in os.listdir(self.dataset_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ])