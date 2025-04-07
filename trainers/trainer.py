import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import os
from typing import Optional, Callable
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import wandb # Import wandb
from torch import nn
from einops import rearrange # Import rearrange from einops

# Assuming ExpertModel and RouterModel are correctly defined
from models.expert import ExpertModel
from models.router import RouterModel

# Placeholder for noise schedule functions (alpha_t, sigma_t)
# These would typically be defined in a separate utility file (e.g., models.util or trainers.schedule)
# Example: Standard linear variance schedule leading to sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod
def get_linear_noise_schedule(timesteps=1000, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# Simplified forward diffusion based on the schedule
def forward_diffuse(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    
    # Ensure t is on the same device as the schedule tensors
    device = sqrt_alphas_cumprod.device
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(x0.device).view(-1, *([1] * (x0.ndim - 1)))
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(x0.device).view(-1, *([1] * (x0.ndim - 1)))

    xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    return xt, noise

# Simplified reverse step to predict x0 from noise prediction (common diffusion objective)
# Note: The blog code example uses a slightly different reverse_diffuse. We'll use noise prediction loss.
# def reverse_diffuse_x0(xt, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, pred_noise):
#     sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(xt.device).view(-1, 1, 1, 1)
#     sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(xt.device).view(-1, 1, 1, 1)
#     x0_pred = (xt - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
#     return x0_pred


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        dataloader: DataLoader,
        device: torch.device,
        lr_scheduler: Optional[_LRScheduler],
        num_train_steps: int,
        gradient_accumulation_steps: int,
        log_frequency: int,
        checkpoint_frequency: int,
        checkpoint_dir: str,
        num_diffusion_timesteps: int,
        beta_start: float,
        beta_end: float,
        use_amp: bool,
        is_distributed: bool,
        is_main_process: bool,
        world_size: int,
        use_wandb: bool = False,
        max_grad_norm: Optional[float] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.num_train_steps = num_train_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_frequency = log_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_dir = checkpoint_dir
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process
        self.world_size = world_size
        self.use_wandb = use_wandb
        self.max_grad_norm = max_grad_norm

        self.scaler = GradScaler(enabled=self.use_amp)

        # Noise schedule setup using configured values
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = get_linear_noise_schedule(
            timesteps=self.num_diffusion_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0

    def _get_raw_model(self):
        from torch.nn.parallel import DistributedDataParallel as DDP 
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save_checkpoint(self, filename: Optional[str] = None):
        """Saves the model, optimizer, and scheduler state (only on main process)."""
        if not self.is_main_process:
            return

        if filename is None:
            model_name = self._get_raw_model().__class__.__name__.lower()
            filename = f"{model_name}_step_{self.global_step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        raw_model = self._get_raw_model()
        model_state_dict = raw_model.state_dict()

        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        try:
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved by Rank {dist.get_rank() if self.is_distributed else 0} to {filepath}")
        except Exception as e:
            print(f"Error saving checkpoint to {filepath}: {e}")

    def load_checkpoint(self, filepath: str):
        """Loads state from a checkpoint (all processes load)."""
        from torch.nn.parallel import DistributedDataParallel as DDP
        if not os.path.exists(filepath):
            print(f"Rank {dist.get_rank() if self.is_distributed else 0} Warning: Checkpoint file not found at {filepath}. Starting from scratch.")
            return

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            raw_model = self._get_raw_model()
            model_state_dict = checkpoint['model_state_dict']
            missing_keys, unexpected_keys = raw_model.load_state_dict(model_state_dict, strict=False)

            if missing_keys: print(f"Rank {dist.get_rank() if self.is_distributed else 0} Warning: Missing keys: {missing_keys}")
            if unexpected_keys: print(f"Rank {dist.get_rank() if self.is_distributed else 0} Warning: Unexpected keys: {unexpected_keys}")

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and self.use_amp:
                 self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.global_step = checkpoint.get('global_step', 0)
            print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Checkpoint loaded. Resuming from step {self.global_step}")
        except Exception as e:
            print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Error loading checkpoint: {e}")
            self.global_step = 0


class ExpertTrainer(BaseTrainer):
    def __init__(self, expert_id: int, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(expert_id, int) or expert_id < 0:
            raise ValueError("ExpertTrainer requires a valid non-negative expert_id.")
        if not isinstance(patch_size, int) or patch_size <= 0:
             raise ValueError("ExpertTrainer requires a valid positive patch_size.")
        self.expert_id = expert_id
        self.patch_size = patch_size
        self.loss_fn = nn.MSELoss() # Standard for diffusion noise prediction

    def _get_batch_data(self, batch):
        """Extracts and prepares data needed for the Expert model from the batch dict."""
        required_keys = {'latents', 'clip', 't5', 'img_ids', 'txt_ids', 'clusters'}
        if not all(key in batch for key in required_keys):
            missing = required_keys - batch.keys()
            raise ValueError(f"Missing required keys in expert batch: {missing}")

        # Extract mandatory data
        x0 = batch['latents'].to(self.device) # The 'clean' latents [B, C, H, W]
        y = batch['clip'].to(self.device)     # CLIP condition [B, CondDim]
        txt = batch['t5'].to(self.device)     # T5 condition [B, SeqLen, T5Dim]
        img_ids = batch['img_ids'].to(self.device) # Image positional IDs [B, NumImgPatches, 3]
        txt_ids = batch['txt_ids'].to(self.device) # Text positional IDs [B, SeqLen, 3]
        # cluster_ids = batch['clusters'].to(self.device) # Cluster IDs [B] - not directly used by expert model forward pass

        # --- Patchify Image Latents ---
        # Assuming ExpertModel underlying Flux expects patches [B, NumPatches, PatchDim]
        try:
            # Use the patch size read from config
            ph = pw = self.patch_size
            # Check if latent dimensions are divisible by patch size
            _, _, H, W = x0.shape
            if H % ph != 0 or W % pw != 0:
                 raise ValueError(f"Latent dimensions ({H}x{W}) not divisible by patch size ({ph}x{pw})")

            # Use einops.rearrange for patchifying
            x0_patched = rearrange(x0, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)

        except ValueError as e:
             # Catch specific divisibility error
             raise RuntimeError(f"Error patchifying latents in ExpertTrainer: {e}. Latent shape: {x0.shape}, Patch size: {self.patch_size}")
        except Exception as e:
             # Catch other rearrange errors
             raise RuntimeError(f"Error patchifying latents in ExpertTrainer._get_batch_data: {e}. Latent shape: {x0.shape}")


        # Generate random timesteps
        t = torch.randint(0, self.num_diffusion_timesteps, (x0.shape[0],), device=self.device).long()

        # Return dictionary matching ExpertModel forward args + necessary extras (x0, t)
        return {
            'img': x0_patched, # Pass patched latents as 'img' to match Flux forward signature
            'img_ids': img_ids,
            'txt': txt,
            'txt_ids': txt_ids,
            'timesteps': t,    # Keep 'timesteps' name for clarity, Flux forward expects this
            'y': y,
            'guidance': None, # Pass None for standard training
            # Keep original x0 for noise generation/loss calculation
            'x0_unpatched': x0, # Store original unpatched latents (renamed for clarity)
            # 't_int': t # Store integer timesteps if needed elsewhere
        }

    def train(self):
        self.model.train()
        log_start_time = time.time()
        total_loss_for_log = 0.0
        total_grad_norm_for_log = 0.0

        print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Starting training for Expert {self.expert_id} from step {self.global_step}...")
        # Ensure dataloader is iterable
        try:
            data_iter = iter(self.dataloader)
        except TypeError:
            print(f"Error: DataLoader for Expert {self.expert_id} is not iterable. Check initialization.")
            return # Cannot train

        while self.global_step < self.num_train_steps:
            accumulated_loss_per_step = 0.0
            # Ensure optimizer is valid before zero_grad
            if self.optimizer is None:
                print(f"Error: Optimizer not initialized for Expert {self.expert_id}.")
                return
            self.optimizer.zero_grad()

            for i in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                    if batch is None: # Handle potential None from collate_fn
                         print(f"Warning: Skipped None batch from dataloader (Rank {dist.get_rank() if self.is_distributed else 0})")
                         continue
                except StopIteration:
                    print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Resetting dataloader iterator.")
                    data_iter = iter(self.dataloader)
                    try:
                        batch = next(data_iter)
                        if batch is None:
                            print(f"Warning: Skipped None batch after iterator reset (Rank {dist.get_rank() if self.is_distributed else 0})")
                            continue
                    except StopIteration:
                        print(f"Error: DataLoader became empty unexpectedly after reset (Rank {dist.get_rank() if self.is_distributed else 0}).")
                        return # Cannot continue if dataloader is persistently empty
                except Exception as e:
                    print(f"Error fetching batch: {e}. Skipping accumulation step.")
                    continue # Skip this gradient accumulation step


                try:
                    batch_data = self._get_batch_data(batch)
                except Exception as e:
                     print(f"Error processing batch data: {e}. Skipping accumulation step.")
                     continue # Skip this step if batch processing fails

                # Extract required tensors, ensure they exist
                x0_unpatched = batch_data.get('x0_unpatched')
                t = batch_data.get('timesteps')
                noise_target = batch_data.get('noise') # Get pre-generated noise if available

                if x0_unpatched is None or t is None:
                     print("Error: Missing 'x0_unpatched' or 'timesteps' in batch_data. Skipping step.")
                     continue
                     
                # --- Forward Diffusion on UNPATCHED latents ---
                # Noise is added to the original latent representation
                xt_unpatched, noise = forward_diffuse(
                    x0_unpatched, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, noise=noise_target
                )
                
                # --- Patch the NOISY latents BEFORE passing to the model ---
                # Model expects patched input corresponding to noisy latent
                try:
                    ph = pw = self.patch_size
                    xt_patched = rearrange(xt_unpatched, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)
                except Exception as e:
                    print(f"Error patching noisy latents xt_unpatched: {e}. Skipping step.")
                    continue
                
                # Prepare model inputs
                # Model forward expects 'img' to be the *patched noisy* input
                model_kwargs = {
                    'img': xt_patched, # Pass patched noisy latents
                    'img_ids': batch_data['img_ids'],
                    'txt': batch_data['txt'],
                    'txt_ids': batch_data['txt_ids'],
                    'timesteps': t, # Pass original timesteps
                    'y': batch_data['y'],
                    'guidance': batch_data['guidance'] # Should be None during training
                }


                with autocast(enabled=self.use_amp):
                    # Pass arguments directly, not nested dict
                    pred_noise_patched = self.model(**model_kwargs) # Model predicts noise in patched format
                    
                    # --- Loss Calculation ---
                    # Target noise should also be in the patched format to match prediction
                    try:
                         ph = pw = self.patch_size
                         noise_patched = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)
                    except Exception as e:
                         print(f"Error patching target noise: {e}. Skipping step.")
                         continue

                    loss = self.loss_fn(pred_noise_patched, noise_patched) # Compare patched prediction with patched noise
                    loss = loss / self.gradient_accumulation_steps

                # Accumulate loss before scaling
                # Detach loss before accumulating to avoid graph buildup if accumulating items
                accumulated_loss_per_step += loss.item() * self.gradient_accumulation_steps # Use item() for accumulation


                # Scaled backward pass
                self.scaler.scale(loss).backward()


            # Unscale, clip, step, update scaler (outside accumulation loop)
            try:
                self.scaler.unscale_(self.optimizer)
            except Exception as e:
                 print(f"Error unscaling optimizer: {e}")
                 # Potentially skip step if unscale fails catastrophically?

            if self.max_grad_norm is not None:
                 try:
                     # Filter parameters that have gradients before clipping
                     params_to_clip = [p for p in self.model.parameters() if p.grad is not None]
                     if params_to_clip:
                          grad_norm = clip_grad_norm_(params_to_clip, self.max_grad_norm)
                          total_grad_norm_for_log += grad_norm.item()
                     else:
                          grad_norm = 0.0 # Or some placeholder if needed
                 except Exception as e:
                      print(f"Error during gradient clipping: {e}")
                      # Handle error, e.g., log it, maybe skip logging this norm
            else:
                 # Calculate grad norm manually (optional, can be expensive)
                 try:
                     # Filter parameters with gradients
                     params_with_grads = [p for p in self.model.parameters() if p.grad is not None]
                     if params_with_grads:
                         # Use torch.stack for potentially better performance on GPU
                         all_norms = torch.stack([torch.norm(p.grad.detach().float(), 2) for p in params_with_grads])
                         grad_norm = torch.norm(all_norms, 2)
                         total_grad_norm_for_log += grad_norm.item()
                     else:
                          grad_norm = 0.0
                 except RuntimeError as e:
                      # Catch potential errors like "stack expects a non-empty TensorList"
                      print(f"Error calculating manual grad norm (likely no grads found): {e}")
                 except Exception as e:
                      # Catch other unexpected errors
                      print(f"Unexpected error calculating manual grad norm: {e}")


            # Optimizer step and scaler update
            # Check if any gradients were actually computed before stepping
            # A simple check could be if total_grad_norm_for_log increased, or check param.grad directly
            has_grads = any(p.grad is not None for p in self.model.parameters())
            if has_grads:
                 self.scaler.step(self.optimizer)
                 self.scaler.update()
                 # Only step scheduler if optimizer stepped
                 if self.lr_scheduler is not None: 
                      self.lr_scheduler.step()
            else:
                 # If no gradients (e.g., due to skipping all accumulation steps), skip optimizer step
                 # Log this situation?
                 if self.gradient_accumulation_steps > 0: # Avoid logging if accum steps is 0
                      print(f"Warning: Skipping optimizer step {self.global_step + 1} as no gradients were found.")
                 # Need to decide if scheduler should step even if optimizer doesn't.
                 # Usually, scheduler step is tied to optimizer step.
                 # if self.lr_scheduler is not None: self.lr_scheduler.step() # Optional: Step scheduler anyway?


            if self.is_distributed:
                 # Use non-blocking calls and average loss across devices
                 loss_tensor = torch.tensor(accumulated_loss_per_step, device=self.device)
                 dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG, async_op=False) # Sync here for logging
                 avg_step_loss = loss_tensor.item()
            else:
                 avg_step_loss = accumulated_loss_per_step

            total_loss_for_log += avg_step_loss

            # Logging logic (remains the same)
            if self.is_main_process and (self.global_step + 1) % self.log_frequency == 0:
                avg_loss_log_period = total_loss_for_log / self.log_frequency
                avg_grad_norm_log_period = total_grad_norm_for_log / self.log_frequency
                current_time = time.time()
                elapsed_time_log = current_time - log_start_time
                steps_per_sec = self.log_frequency / elapsed_time_log if elapsed_time_log > 0 else 0
                lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0 # Handle empty groups

                print(f"Expert {self.expert_id} | Step: {self.global_step+1}/{self.num_train_steps} | "
                      f"Avg Loss: {avg_loss_log_period:.4f} | Grad Norm: {avg_grad_norm_log_period:.4f} | LR: {lr:.6f} | "
                      f"Steps/sec: {steps_per_sec:.2f}")

                if self.use_wandb:
                    log_data = {
                        f"expert_{self.expert_id}/loss": avg_loss_log_period,
                        f"expert_{self.expert_id}/grad_norm": avg_grad_norm_log_period,
                        "train/learning_rate": lr,
                        "train/steps_per_second": steps_per_sec,
                        "train/step": self.global_step + 1,
                        "expert_id": self.expert_id
                    }
                    wandb.log(log_data, step=self.global_step + 1)

                total_loss_for_log = 0.0
                total_grad_norm_for_log = 0.0
                log_start_time = current_time


            # Checkpointing logic (remains the same)
            if (self.global_step + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()

            self.global_step += 1
            if self.global_step >= self.num_train_steps:
                 break

        # Final checkpoint save (remains the same)
        self.save_checkpoint(filename=f"expert_{self.expert_id}_final.pt")
        print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Training finished for Expert {self.expert_id}.")


class RouterTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_batch_data(self, batch):
        """Extracts x0, condition (y), and cluster_idx for the Router."""
        if not isinstance(batch, dict):
             raise TypeError(f"Expected batch to be a dict, but got {type(batch)}")

        # --- Extract mandatory data ---
        x0 = batch.get('image')
        cluster_idx = batch.get('cluster_idx')

        if x0 is None:
            raise ValueError("Batch dictionary missing mandatory 'image' key (latent features).")
        if cluster_idx is None:
            raise ValueError("Batch dictionary missing mandatory 'cluster_idx' key.")

        x0 = x0.to(self.device)
        # Ensure cluster_idx is LongTensor on the correct device
        if isinstance(cluster_idx, (int, float)): # Handle potential scalar from dataset
             cluster_idx = torch.tensor([cluster_idx] * x0.shape[0], device=self.device).long()
        else:
             cluster_idx = cluster_idx.to(self.device).long()

        # --- Extract optional condition 'y' ---
        condition_y = None
        if self._get_raw_model().has_cond: # Check if the router expects 'y'
            condition_y = batch.get('y')
            if condition_y is None:
                 raise ValueError("RouterModel expects condition 'y', but it was not found in the batch dictionary.")
            condition_y = condition_y.to(self.device)

        return x0, condition_y, cluster_idx # Return image, optional condition 'y', and cluster index

    def train(self):
        self.model.train()
        log_start_time = time.time()
        total_loss_for_log = 0.0
        total_grad_norm_for_log = 0.0

        print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Starting training for Router from step {self.global_step}...")
        data_iter = iter(self.dataloader)

        while self.global_step < self.num_train_steps:
            accumulated_loss_per_step = 0.0
            self.optimizer.zero_grad()

            for i in range(self.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                x0, condition_y, cluster_idx = self._get_batch_data(batch)
                B = x0.shape[0]
                t = torch.randint(0, self.num_diffusion_timesteps, (B,), device=self.device).long()
                xt, _ = forward_diffuse(
                    x0, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod
                )

                with autocast(enabled=self.use_amp):
                    model_kwargs = {'x': xt, 't': t.float()}
                    if condition_y is not None: model_kwargs['y'] = condition_y
                    logits = self.model(**model_kwargs)
                    loss = F.cross_entropy(logits, cluster_idx)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accumulated_loss_per_step += loss.item() * self.gradient_accumulation_steps

            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm is not None:
                 grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                 total_grad_norm_for_log += grad_norm.item()
            else:
                 try:
                     grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if p.grad is not None]), 2)
                     total_grad_norm_for_log += grad_norm.item()
                 except:
                      pass

            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.lr_scheduler is not None: self.lr_scheduler.step()

            if self.is_distributed:
                 loss_tensor = torch.tensor(accumulated_loss_per_step, device=self.device)
                 dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                 avg_step_loss = loss_tensor.item()
            else:
                 avg_step_loss = accumulated_loss_per_step

            total_loss_for_log += avg_step_loss

            if self.is_main_process and (self.global_step + 1) % self.log_frequency == 0:
                avg_loss_log_period = total_loss_for_log / self.log_frequency
                avg_grad_norm_log_period = total_grad_norm_for_log / self.log_frequency
                current_time = time.time()
                elapsed_time_log = current_time - log_start_time
                steps_per_sec = self.log_frequency / elapsed_time_log if elapsed_time_log > 0 else 0
                lr = self.optimizer.param_groups[0]['lr']

                print(f"Router | Step: {self.global_step+1}/{self.num_train_steps} | "
                      f"Avg Loss: {avg_loss_log_period:.4f} | Grad Norm: {avg_grad_norm_log_period:.4f} | LR: {lr:.6f} | "
                      f"Steps/sec: {steps_per_sec:.2f}")

                if self.use_wandb:
                    log_data = {
                        "router/loss": avg_loss_log_period,
                        "router/grad_norm": avg_grad_norm_log_period,
                        "train/learning_rate": lr,
                        "train/steps_per_second": steps_per_sec,
                        "train/step": self.global_step + 1
                    }
                    wandb.log(log_data, step=self.global_step + 1)

                total_loss_for_log = 0.0
                total_grad_norm_for_log = 0.0
                log_start_time = current_time

            if (self.global_step + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()

            self.global_step += 1
            if self.global_step >= self.num_train_steps:
                 break

        self.save_checkpoint(filename="router_final.pt")
        print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Training finished for Router.")
