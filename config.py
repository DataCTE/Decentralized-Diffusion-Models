"""Configuration for Decentralized Diffusion Models."""

class DDMConfig:
    """Configuration for Decentralized Diffusion Models"""
    def __init__(self):
        # Model architecture
        self.hidden_dim = 1152  # DiT-XL dimension
        self.num_layers = 28     # DiT-XL depth
        self.num_heads = 16      # DiT-XL heads
        self.ffn_dim = 3072      # DiT-XL MLP ratio
        self.batch_size = 16
        
        # DDM specific settings
        self.num_experts = 8     # Number of expert models (paper recommends 8)
        self.router_hidden_dim = 512  # Smaller router dimension
        
        # Training settings
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.adam_betas = (0.9, 0.99)
        
        # Data settings
        self.patch_size = 32
        self.image_size = 512
        self.dataset_path = "/home/alex/workspace/datasets/danbooru2025"  # Path to training dataset
        self.num_workers = 2     # DataLoader workers
        self.pin_memory = True   # Pin memory for faster data transfer
        
        # Training settings
        self.num_steps = 400_000  
        self.log_dir = "runs/ddm"
        self.save_dir = "checkpoints/ddm"
        self.checkpoint_dir = "checkpoints/ddm"  # Added to match save_dir
        self.sample_dir = "samples/ddm"  # Directory to save generated samples
        
        # Inference settings
        self.use_top_k = 1  # Number of experts to use at inference time
        self.inference_steps = 50  # Number of steps for sampling
        self.inference_prompt = "1girl"  # Default prompt
        
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
        
        # Paper-recommended hyperparameters
        self.expert_batch_size = 4  # Per-expert batch size
        self.router_learning_rate = 3e-5   # Different from expert LR
        self.recluster_epochs = 3          # Paper recommends 3 passes
        self.ema_decay = 0.9999            # For model stability
        self.max_grad_norm = 1.0           # Gradient clipping
        self.fid_sample_size = 50000       # For meaningful FID
        self.inception_metrics_interval = 10000  # Steps between eval
        self.diversity_threshold = 0.75    # Novelty measurement
        
        # Feature extraction settings
        self.feature_batch_size = 64
        self.feature_workers = 4
        self.dino_size = 518  # Size for DINO feature extraction
        
        # Distillation settings
        self.distill_lr = 1e-5
        self.distill_batch_size = 4
        self.distill_epochs = 10
        self.distill_samples = 10000  # Number of samples for distillation 