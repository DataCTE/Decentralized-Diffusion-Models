import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler # Example common scheduler provider
import os
import random
import numpy as np
import fire # For easy CLI argument parsing
import toml # Import the TOML library
from types import SimpleNamespace # To convert dict to object-like structure
from typing import Optional # Added Optional
import wandb # Import wandb

# Project imports - adjust paths if necessary
# from config import get_config, Config # Remove old config import
from models.expert import ExpertModel
from models.router import RouterModel
from trainers.trainer import ExpertTrainer, RouterTrainer
# Assuming DDMDataset and BucketBatchSampler are correctly implemented
# Need to ensure DDMDataset is adapted for router/expert data loading
from data.dataset import DDMDataset, BucketBatchSampler
from utils import dict_to_sns, find_latest_checkpoint # Import helpers from utils
from models.flux.model import FluxParams # Import FluxParams

def setup_distributed():
    """Initializes torch.distributed"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Initializing distributed training: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        # Ensure backend is explicitly set if needed, nccl is common for NVIDIA GPUs
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
             torch.cuda.set_device(local_rank)
             device = torch.device(f"cuda:{local_rank}")
        else:
             device = torch.device("cpu") # Fallback for CPU-only distributed (less common)
        return rank, world_size, local_rank, device
    else:
        print("Not running in distributed mode.")
        # Setup for single GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device # rank, world_size, local_rank, device

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(config_path: str = "config.toml", **kwargs): # Default to config.toml
    """
    Main training function.

    Args:
        config_path (str): Path to the TOML configuration file.
        **kwargs: Command-line overrides for configuration parameters (e.g., train.batch_size=128).
                  Overrides values loaded from the TOML file.
    """
    # --- 1. Load Configuration ---
    print(f"Loading configuration from: {config_path}")
    try:
        config_dict = toml.load(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading TOML config file: {e}")
        return

    # Command-line overrides are handled by editing the config.toml file directly for simplicity.

    # Convert dict to SimpleNamespace for easier access (cfg.train.batch_size)
    cfg = dict_to_sns(config_dict)

    # Derive checkpoint_dir after loading
    # Ensure train section exists before accessing checkpoint_dir
    if not hasattr(cfg, 'train'): raise ValueError("Config missing [train] section.")
    if not hasattr(cfg.train, 'output_dir'): raise ValueError("Config missing output_dir in [train] section.")
    cfg.train.checkpoint_dir = os.path.join(cfg.train.output_dir, "checkpoints")

    # --- Validate presence of essential [train] config values ---
    if not hasattr(cfg, 'train'):
        raise ValueError("Configuration file missing [train] section.")

    required_train_keys = [
        'output_dir', 'model_type', 'batch_size', 'num_train_steps',
        'gradient_accumulation_steps',
        'learning_rate', 'adam_beta1', 'adam_beta2', 'adam_weight_decay', 'adam_epsilon', # <-- Added optimizer keys
        'lr_scheduler_type',
        'log_frequency', 'checkpoint_frequency', 'num_diffusion_timesteps',
        'beta_start', 'beta_end', 'distributed', 'use_mixed_precision'
        # Add other absolutely essential keys if needed
    ]
    missing_keys = [key for key in required_train_keys if not hasattr(cfg.train, key)]
    if missing_keys:
        raise ValueError(f"Missing required keys in [train] section of config: {missing_keys}")

    # Specific check for expert_id if type is expert
    if cfg.train.model_type == "expert" and not hasattr(cfg.train, 'expert_id'):
         raise ValueError("Missing required key 'expert_id' in [train] section when model_type is 'expert'.")

    # --- 2. Setup Environment ---
    rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)
    is_distributed = world_size > 1 and getattr(cfg.train, 'distributed', False) # Check distributed flag safely
    set_seed(getattr(cfg.train, 'seed', 42) + rank) # Use getattr for seed

    if is_main:
        print("-------------------- Configuration --------------------")
        # Print the original dictionary for clarity before conversion
        print(config_dict)
        print("-------------------------------------------------------")
        # Ensure output_dir exists before creating subdirs
        os.makedirs(cfg.train.output_dir, exist_ok=True)
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        # Make specific subdirs for router/experts later when trainer is initialized

        # Initialize wandb
        # Determine run name based on model type and expert ID
        model_type = getattr(cfg.train, 'model_type', 'unknown')
        run_name = f"{model_type}"
        if model_type == "expert":
            # Use getattr for safe access to expert_id
            expert_id_val = getattr(cfg.train, 'expert_id', 'unknown')
            run_name += f"_{expert_id_val}"
        
        try:
             # Check if wandb is enabled in config (add this if desired)
             use_wandb = getattr(cfg.train, 'use_wandb', True) 
             if use_wandb:
                 wandb.init(
                      project=getattr(cfg.train, 'wandb_project', "decentralized-diffusion"), # Configurable project
                      name=run_name,
                      config=config_dict, # Log the raw config dictionary
                      dir=cfg.train.output_dir # Optional: Set wandb local log directory
                 )
                 print("WandB initialized successfully.")
             else:
                  print("WandB disabled by config.")
                  wandb.init(mode="disabled")

        except Exception as e:
             print(f"Error initializing WandB: {e}. Proceeding without WandB logging.")
             wandb.init(mode="disabled") # Disable wandb if init fails


    # --- 3. Initialize Dataset and Dataloader ---
    # Convert relevant parts of the main config to a dict for DDMDataset
    # Pass a FLAT dictionary, not the nested cfg
    dataset_config_dict = {
        # Extract values safely using getattr from cfg.data if it exists
        'feature_cache_path': getattr(cfg.data, 'feature_cache_path', None),
        'latent_channels': getattr(cfg.data, 'latent_channels', None),
        # num_clusters is needed by the dataset to determine which expert a sample belongs to
        'num_experts': getattr(cfg.model, 'num_clusters', None), 
        # Bucket settings need to be directly accessible
        'bucket_thresholds': getattr(cfg.data, 'bucket_thresholds', {}),
        'bucket_scale': getattr(cfg.data, 'bucket_scale', 8),
        'buckets': getattr(cfg.data, 'buckets', []),
    }
    # Validate required dataset config keys
    if not dataset_config_dict['feature_cache_path']: raise ValueError("Missing feature_cache_path in [data] config.")
    if not dataset_config_dict['num_experts']: raise ValueError("Missing num_clusters in [model] config (needed by dataset).")

    print(f"Rank {rank}: Initializing DDMDataset...")
    # Pass the flat dict
    dataset = DDMDataset(config_dict=dataset_config_dict)
    print(f"Rank {rank}: DDMDataset initialized with {len(dataset)} samples.")


    # --- Modify Sampler for DDP ---
    # Use DistributedSampler to wrap the dataset indices OR adapt BucketBatchSampler
    # Using DistributedSampler is simpler if bucketing isn't strictly required per batch *during DDP*
    # If bucketing IS required per batch with DDP, BucketBatchSampler needs adaptation (shown below)
    print(f"Rank {rank}: Initializing Sampler (Distributed={is_distributed})...")
    if is_distributed:
         # Option B: Adapt BucketBatchSampler for DDP (keeps buckets per batch)
         batch_sampler = BucketBatchSampler(
              dataset=dataset, 
              batch_size=cfg.train.batch_size, 
              shuffle=True, 
              drop_last=True,
              logger=dataset.logger # Pass logger
         )
    else:
         # Non-distributed: use BucketBatchSampler directly
         batch_sampler = BucketBatchSampler(
              dataset=dataset, 
              batch_size=cfg.train.batch_size, 
              shuffle=True, 
              drop_last=True,
              logger=dataset.logger # Pass logger
         )


    print(f"Rank {rank}: Initializing DataLoader...")
    # Ensure collate_fn is correctly referenced if it's a static method
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler, # Use the (potentially DDP-aware) BucketBatchSampler
        num_workers=getattr(cfg.train, 'num_workers', 4), # Get num_workers safely
        pin_memory=True, # Important for performance
        collate_fn=DDMDataset.collate_fn # Use the static collate function
    )
    print(f"Rank {rank}: DataLoader initialized with {len(dataloader)} batches for this rank.")

    # --- 4. Initialize Model ---
    model = None
    trainer_class = None
    trainer_checkpoint_subdir = ""
    expert_patch_size_val = None

    if cfg.train.model_type == "expert":
        expert_id = getattr(cfg.train, 'expert_id', None)
        if expert_id is None or not isinstance(expert_id, int) or expert_id < 0:
             raise ValueError("Expert training requires a valid non-negative integer 'expert_id' in [train] config.")
        print(f"Rank {rank}: Initializing ExpertModel {expert_id}...")

        # --- Validate Expert Config ---
        if not hasattr(cfg, 'model'):
            raise ValueError("Config missing [model] section for expert parameters.")
        required_expert_keys = [
            'expert_patch_size', 'expert_in_channels', 'expert_out_channels',
            'expert_vec_in_dim', 'expert_context_in_dim', 'expert_hidden_size',
            'expert_mlp_ratio', 'expert_num_heads', 'expert_depth',
            'expert_depth_single_blocks', 'expert_axes_dim', 'expert_theta',
            'expert_qkv_bias', 'expert_guidance_embed'
            # Add num_clusters if needed by ExpertModel itself, though usually just for dataset/router
        ]
        missing_expert_keys = [key for key in required_expert_keys if not hasattr(cfg.model, key)]
        if missing_expert_keys:
             raise ValueError(f"Missing required keys in [model] section for expert: {missing_expert_keys}")

        expert_raw_config = {}
        prefix = "expert_"
        expert_patch_size_val = cfg.model.expert_patch_size # Read directly after validation

        for k in dir(cfg.model):
            if k.startswith(prefix):
                v = getattr(cfg.model, k)
                if k != 'expert_patch_size':
                    new_key = k[len(prefix):]
                    expert_raw_config[new_key] = v

        try:
            flux_params_for_expert = FluxParams(**expert_raw_config)
            model = ExpertModel(mmdit_params=flux_params_for_expert)
        except Exception as e:
             print(f"Error creating ExpertModel: {e}")
             raise e

        trainer_class = ExpertTrainer
        trainer_checkpoint_subdir = f"expert_{expert_id}"

    elif cfg.train.model_type == "router":
        print(f"Rank {rank}: Initializing RouterModel...")
        # --- Validate Router Config ---
        if not hasattr(cfg, 'model'):
            raise ValueError("Config missing [model] section for router parameters.")
        required_router_keys = [
            'num_clusters', 'router_input_size', 'router_patch_size',
            'router_in_channels', 'router_hidden_size', 'router_depth',
            'router_num_heads', 'router_mlp_ratio'
            # router_cond_dim is optional
        ]
        missing_router_keys = [key for key in required_router_keys if not hasattr(cfg.model, key)]
        if missing_router_keys:
             raise ValueError(f"Missing required keys in [model] section for router: {missing_router_keys}")

        # Extract router-specific config params directly after validation
        router_cond_dim = getattr(cfg.model, 'router_cond_dim', None) # Keep default for optional

        model = RouterModel(
            # Access required attributes directly
            num_clusters=cfg.model.num_clusters,
            input_size=cfg.model.router_input_size,
            patch_size=cfg.model.router_patch_size,
            in_channels=cfg.model.router_in_channels,
            hidden_size=cfg.model.router_hidden_size,
            depth=cfg.model.router_depth,
            num_heads=cfg.model.router_num_heads,
            mlp_ratio=cfg.model.router_mlp_ratio,
            cond_dim=router_cond_dim, # Pass the potentially None value
        )

        trainer_class = RouterTrainer
        trainer_checkpoint_subdir = "router"

    else:
        raise ValueError(f"Unknown model_type in config: {cfg.train.model_type}. Choose 'expert' or 'router'.")

    if model is None:
         raise RuntimeError("Model initialization failed.") # Should not happen if logic above is correct

    model = model.to(device)
    if is_main: print(f"Model {cfg.train.model_type} initialized on {device}.")

    # --- Wrap Model with DDP if distributed ---
    if is_distributed:
        print(f"Rank {rank}: Wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                    output_device=local_rank if device.type == 'cuda' else None,
                    find_unused_parameters=False) # Set to True if needed
        print(f"Rank {rank}: Model wrapped with DDP.")

    # --- 5. Initialize Optimizer and Scheduler ---
    print(f"Rank {rank}: Initializing Optimizer and Scheduler...") # Added print
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,              # <-- Direct access
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2), # <-- Direct access
        weight_decay=cfg.train.adam_weight_decay, # <-- Direct access
        eps=cfg.train.adam_epsilon,               # <-- Direct access
    )

    # Ensure gradient accumulation steps >= 1
    grad_accum_steps = max(1, cfg.train.gradient_accumulation_steps) # Read directly after validation

    lr_scheduler = get_scheduler(
        name=cfg.train.lr_scheduler_type, # Read directly
        optimizer=optimizer,
        num_warmup_steps=getattr(cfg.train, 'lr_warmup_steps', 0) * grad_accum_steps, # Keep default for optional warmup
        num_training_steps=cfg.train.num_train_steps * grad_accum_steps, # Read directly
    )
    print(f"Rank {rank}: Optimizer and Scheduler initialized.") # Added print

    # --- 6. Initialize Trainer ---
    print(f"Rank {rank}: Initializing {trainer_class.__name__}...")
    specific_checkpoint_dir = os.path.join(cfg.train.checkpoint_dir, trainer_checkpoint_subdir)
    if is_main:
         os.makedirs(specific_checkpoint_dir, exist_ok=True)

    # Common kwargs - Now directly accessing cfg.train attributes after validation
    # Removed default fallbacks for required parameters
    common_trainer_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'dataloader': dataloader,
        'device': device,
        'lr_scheduler': lr_scheduler,
        'num_train_steps': cfg.train.num_train_steps,
        'gradient_accumulation_steps': grad_accum_steps,
        'log_frequency': cfg.train.log_frequency,
        'checkpoint_frequency': cfg.train.checkpoint_frequency,
        'checkpoint_dir': specific_checkpoint_dir,
        'num_diffusion_timesteps': cfg.train.num_diffusion_timesteps,
        'beta_start': cfg.train.beta_start,
        'beta_end': cfg.train.beta_end,
        'use_amp': cfg.train.use_mixed_precision,
        'is_distributed': is_distributed,
        'is_main_process': is_main,
        'world_size': world_size,
        # Keep getattr with default for optional parameters:
        'use_wandb': is_main and getattr(cfg.train, 'use_wandb', False) and wandb.run is not None and wandb.run.mode != "disabled",
        'max_grad_norm': getattr(cfg.train, 'max_grad_norm', None)
    }

    # Add model-specific kwargs
    trainer_init_kwargs = common_trainer_kwargs.copy()
    if cfg.train.model_type == "expert":
        trainer_init_kwargs['expert_id'] = cfg.train.expert_id # Read directly
        trainer_init_kwargs['patch_size'] = expert_patch_size_val
    # No specific kwargs needed for router

    trainer = trainer_class(**trainer_init_kwargs)
    print(f"Rank {rank}: Trainer initialized.")

    # --- Checkpoint Loading ---
    latest_checkpoint = find_latest_checkpoint(trainer.checkpoint_dir) # find_latest_checkpoint uses the specific dir
    if latest_checkpoint:
         print(f"Rank {rank}: Found latest checkpoint: {latest_checkpoint}. Attempting to load...")
         try:
              trainer.load_checkpoint(latest_checkpoint)
         except Exception as e:
              print(f"Rank {rank}: Failed to load checkpoint {latest_checkpoint}. Starting training from scratch. Error: {e}")
              trainer.global_step = 0
    else:
         print(f"Rank {rank}: No checkpoint found in {trainer.checkpoint_dir}. Starting training from scratch.")

    # --- Synchronization Point before Training ---
    if is_distributed:
        print(f"Rank {rank}: Waiting on barrier before starting training...")
        dist.barrier()
        print(f"Rank {rank}: Barrier passed, starting training.")

    # --- 7. Start Training ---
    print(f"Rank {rank}: Starting training from step {trainer.global_step}...")
    trainer.train() # train() method in trainer should use self.global_step

    # --- 8. Cleanup ---
    if is_main and wandb.run is not None and wandb.run.mode != "disabled":
        wandb.finish() # Finish the wandb run on the main process

    if is_distributed:
        # Ensure barrier happens even if training fails on some ranks
        try:
             dist.barrier() # Wait for all processes to finish training
        except Exception as e:
             print(f"Rank {rank}: Error during final barrier: {e}")
        finally:
             # Always attempt to destroy the process group
             if dist.is_initialized():
                  dist.destroy_process_group()

    print(f"Rank {rank}: Training finished.")


if __name__ == "__main__":
    fire.Fire(train) # Expose the train function to the command line
