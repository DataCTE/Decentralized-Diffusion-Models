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
    cfg.train.checkpoint_dir = os.path.join(cfg.train.output_dir, "checkpoints")


    # --- 2. Setup Environment ---
    rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)
    is_distributed = world_size > 1 and cfg.train.distributed # Check if distributed is enabled and active
    set_seed(cfg.train.seed + rank) # Add rank for different seeds per process

    if is_main:
        print("-------------------- Configuration --------------------")
        # Basic config print - consider using pprint or a dedicated library for complex configs
        # Convert back to dict for printing or use a recursive print function
        print(config_dict)
        print("-------------------------------------------------------")
        os.makedirs(cfg.train.output_dir, exist_ok=True)
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
        # Make specific subdirs for router/experts later when trainer is initialized

        # Initialize wandb
        # Determine run name based on model type and expert ID
        run_name = f"{cfg.train.model_type}"
        if cfg.train.model_type == "expert":
            run_name += f"_{getattr(cfg.train, 'expert_id', 'unknown')}"
        
        try:
             wandb.init(
                  project="decentralized-diffusion", # Or your preferred project name
                  name=run_name,
                  config=config_dict, # Log the raw config dictionary
                  dir=cfg.train.output_dir # Optional: Set wandb local log directory
             )
             print("WandB initialized successfully.")
        except Exception as e:
             print(f"Error initializing WandB: {e}. Proceeding without WandB logging.")
             wandb.init(mode="disabled") # Disable wandb if init fails


    # --- 3. Initialize Dataset and Dataloader ---
    # Convert relevant parts of the main config to a dict for DDMDataset
    # Now directly use the cfg object loaded from TOML
    dataset_config_dict = {
        'feature_cache_path': cfg.data.feature_cache_path,
        'latent_channels': cfg.data.latent_channels,
        'num_experts': cfg.model.num_clusters,
        # Add bucketing parameters if defined in config.toml
        'bucket_thresholds': getattr(cfg.data, 'bucket_thresholds', {}),
        'bucket_scale': getattr(cfg.data, 'bucket_scale', 8),
        'buckets': getattr(cfg.data, 'buckets', []),
        # Add any other parameters DDMDataset expects from its config_dict
    }
    print(f"Rank {rank}: Initializing DDMDataset...")
    # Pass the dict, not the SimpleNamespace directly if DDMDataset expects dict
    dataset = DDMDataset(config_dict=dataset_config_dict)

    # --- Modify Sampler for DDP ---
    # Use DistributedSampler to wrap the dataset indices OR adapt BucketBatchSampler
    # Using DistributedSampler is simpler if bucketing isn't strictly required per batch *during DDP*
    # If bucketing IS required per batch with DDP, BucketBatchSampler needs adaptation (shown below)
    print(f"Rank {rank}: Initializing Sampler (Distributed={is_distributed})...")
    if is_distributed:
         # Option A: Standard Distributed Sampler (ignores buckets per batch)
         # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.train.seed)
         # dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, sampler=sampler, ...)

         # Option B: Adapt BucketBatchSampler for DDP (keeps buckets per batch)
         # Requires BucketBatchSampler to handle DDP logic internally (as implemented in previous step)
         batch_sampler = BucketBatchSampler(
              dataset=dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True
              # BucketBatchSampler should already handle rank/world_size internally
         )
    else:
         # Non-distributed: use BucketBatchSampler directly
         batch_sampler = BucketBatchSampler(
              dataset=dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True
         )


    print(f"Rank {rank}: Initializing DataLoader...")
    # Ensure collate_fn is correctly referenced if it's a static method
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler, # Use the (potentially DDP-aware) BucketBatchSampler
        num_workers=cfg.train.num_workers,
        pin_memory=True, # Important for performance
        collate_fn=DDMDataset.collate_fn # Use the static collate function
    )
    print(f"Rank {rank}: DataLoader initialized.")

    # --- 4. Initialize Model ---
    model = None
    trainer_class = None
    trainer_checkpoint_subdir = "" # To store the specific subdir for checkpoint loading

    if cfg.train.model_type == "expert":
        expert_id = getattr(cfg.train, 'expert_id', None) # Use getattr for safety
        if expert_id is None or expert_id < 0:
             raise ValueError("Expert training requires a valid non-negative 'expert_id' in config.")
        print(f"Rank {rank}: Initializing ExpertModel {expert_id}...")
        # Extract expert-specific config params into a dict from cfg.model
        expert_mmdit_config = {k: getattr(cfg.model, k) for k in dir(cfg.model) if k.startswith('expert_')}
        model = ExpertModel(mmdit_config=expert_mmdit_config)
        trainer_class = ExpertTrainer
        trainer_checkpoint_subdir = f"expert_{expert_id}"
        trainer_init_kwargs = {
             'model': model, 'optimizer': optimizer, 'dataloader': dataloader,
             'device': device, 'lr_scheduler': lr_scheduler,
             'num_train_steps': cfg.train.num_train_steps,
             'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps,
             'log_frequency': cfg.train.log_frequency,
             'checkpoint_frequency': cfg.train.checkpoint_frequency,
             'checkpoint_dir': os.path.join(cfg.train.checkpoint_dir, trainer_checkpoint_subdir),
             'num_diffusion_timesteps': cfg.train.num_diffusion_timesteps,
             'expert_id': expert_id,
             'use_amp': cfg.train.use_mixed_precision,
             'is_distributed': is_distributed,
             'is_main_process': is_main,
             'world_size': world_size,
             'use_wandb': is_main and wandb.run is not None and wandb.run.mode != "disabled",
             'max_grad_norm': getattr(cfg.train, 'max_grad_norm', None)
        }

    elif cfg.train.model_type == "router":
        print(f"Rank {rank}: Initializing RouterModel...")
        # Extract router-specific config params
        router_cond_dim = getattr(cfg.model, 'router_cond_dim', None) # Handle optional attribute
        model = RouterModel(
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
        trainer_init_kwargs = {
             'model': model, 'optimizer': optimizer, 'dataloader': dataloader,
             'device': device, 'lr_scheduler': lr_scheduler,
             'num_train_steps': cfg.train.num_train_steps,
             'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps,
             'log_frequency': cfg.train.log_frequency,
             'checkpoint_frequency': cfg.train.checkpoint_frequency,
             'checkpoint_dir': os.path.join(cfg.train.checkpoint_dir, trainer_checkpoint_subdir),
             'num_diffusion_timesteps': cfg.train.num_diffusion_timesteps,
             'use_amp': cfg.train.use_mixed_precision,
             'is_distributed': is_distributed,
             'is_main_process': is_main,
             'world_size': world_size,
             'use_wandb': is_main and wandb.run is not None and wandb.run.mode != "disabled",
             'max_grad_norm': getattr(cfg.train, 'max_grad_norm', None)
        }

    else:
        raise ValueError(f"Unknown model_type in config: {cfg.train.model_type}. Choose 'expert' or 'router'.")

    model = model.to(device)
    # DDP wrapping below handles distributed training. FSDP can be integrated later if needed.

    # --- Wrap Model with DDP if distributed ---
    if is_distributed:
        print(f"Rank {rank}: Wrapping model with DDP...")
        # find_unused_parameters can be True if some outputs aren't used in loss
        # (might happen with complex models, start with False)
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                    output_device=local_rank if device.type == 'cuda' else None,
                    find_unused_parameters=False) # Set to True if needed
        print(f"Rank {rank}: Model wrapped with DDP.")
        # FSDP Note: If using FSDP, the wrapping happens here instead of DDP
        # from utils.fsdp import create_fsdp_model # Example
        # model = create_fsdp_model(model, cfg, rank=local_rank)

    # --- 5. Initialize Optimizer and Scheduler ---
    # Standard AdamW initialization. Parameter groups can be added later for refinement.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        weight_decay=cfg.train.adam_weight_decay,
        eps=cfg.train.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        name=cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps * cfg.train.gradient_accumulation_steps,
        num_training_steps=cfg.train.num_train_steps * cfg.train.gradient_accumulation_steps,
    )

    # --- 6. Initialize Trainer ---
    print(f"Rank {rank}: Initializing {trainer_class.__name__}...")
    # Construct the specific checkpoint directory for this trainer instance
    specific_checkpoint_dir = os.path.join(cfg.train.checkpoint_dir, trainer_checkpoint_subdir)
    if is_main: # Only main process creates directories
         os.makedirs(specific_checkpoint_dir, exist_ok=True)

    # Common kwargs first
    common_trainer_kwargs = {
        'model': model, 'optimizer': optimizer, 'dataloader': dataloader,
        'device': device, 'lr_scheduler': lr_scheduler,
        'num_train_steps': cfg.train.num_train_steps,
        'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps,
        'log_frequency': cfg.train.log_frequency,
        'checkpoint_frequency': cfg.train.checkpoint_frequency,
        'checkpoint_dir': specific_checkpoint_dir, # Use the specific dir here
        'num_diffusion_timesteps': cfg.train.num_diffusion_timesteps,
        'use_amp': cfg.train.use_mixed_precision,
        'is_distributed': is_distributed,
        'is_main_process': is_main,
        'world_size': world_size,
        'use_wandb': is_main and wandb.run is not None and wandb.run.mode != "disabled",
        'max_grad_norm': getattr(cfg.train, 'max_grad_norm', None) # Get from config or default to None
    }

    # Add model-specific kwargs
    trainer_init_kwargs = common_trainer_kwargs.copy()
    if cfg.train.model_type == "expert":
        trainer_init_kwargs['expert_id'] = expert_id
    # No specific kwargs needed for router beyond common ones currently

    trainer = trainer_class(**trainer_init_kwargs)
    print(f"Rank {rank}: Trainer initialized.")

    # --- Checkpoint Loading ---
    # Find the latest checkpoint in the specific directory for this trainer
    latest_checkpoint = find_latest_checkpoint(trainer.checkpoint_dir)
    if latest_checkpoint:
         print(f"Rank {rank}: Found latest checkpoint: {latest_checkpoint}. Attempting to load...")
         try:
              # Pass is_distributed status to load_checkpoint
              trainer.load_checkpoint(latest_checkpoint)
         except Exception as e:
              print(f"Rank {rank}: Failed to load checkpoint {latest_checkpoint}. Starting training from scratch. Error: {e}")
              # Optionally reset global_step if loading fails partway
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
        dist.barrier() # Wait for all processes to finish training
        dist.destroy_process_group()

    print(f"Rank {rank}: Training finished.")


if __name__ == "__main__":
    fire.Fire(train) # Expose the train function to the command line
