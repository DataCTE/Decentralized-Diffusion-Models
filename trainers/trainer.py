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
        use_amp: bool,
        is_distributed: bool,
        is_main_process: bool,
        world_size: int,
        use_wandb: bool = False, # Add use_wandb flag
        max_grad_norm: Optional[float] = None # Add max_grad_norm option
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
        self.use_wandb = use_wandb # Store the flag
        self.max_grad_norm = max_grad_norm # Store max_grad_norm

        self.scaler = GradScaler(enabled=self.use_amp)

        # Noise schedule setup
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = get_linear_noise_schedule(
            timesteps=self.num_diffusion_timesteps
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
    def __init__(self, expert_id: int, **kwargs):
        super().__init__(**kwargs)
        self.expert_id = expert_id

    def _get_batch_data(self, batch):
        """Extracts relevant data from the collated batch dictionary for the Expert."""
        if not isinstance(batch, dict):
             raise TypeError(f"Expected batch to be a dict, but got {type(batch)}")

        # --- Extract mandatory data ---
        x0 = batch.get('image')
        if x0 is None:
             raise ValueError("Batch dictionary missing mandatory 'image' key (latent features).")
        x0 = x0.to(self.device)

        # --- Extract conditioning data required by Flux/ExpertModel ---
        # Flux expects 'y' (vector), 'txt' (sequence), 'img_ids', 'txt_ids'
        condition = {}
        required_keys = ['y', 'txt', 'img_ids', 'txt_ids'] # Adjust based on exact Flux forward signature
        optional_keys = [] # Add any optional keys Flux might use

        for key in required_keys + optional_keys:
             val = batch.get(key)
             if val is not None:
                 condition[key] = val.to(self.device) if isinstance(val, torch.Tensor) else val
             elif key in required_keys:
                  # This indicates a potential problem in dataset preparation or config mismatch
                  raise ValueError(f"Batch dictionary missing required condition key '{key}' for ExpertModel.")
                  
        return x0, condition # Return image and condition dict

    def train(self):
        self.model.train()
        log_start_time = time.time()
        total_loss_for_log = 0.0
        total_grad_norm_for_log = 0.0

        print(f"Rank {dist.get_rank() if self.is_distributed else 0}: Starting training for Expert {self.expert_id} from step {self.global_step}...")
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

                x0, condition = self._get_batch_data(batch)
                B = x0.shape[0]
                t = torch.randint(0, self.num_diffusion_timesteps, (B,), device=self.device).long()
                xt, noise = forward_diffuse(
                    x0, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod
                )

                with autocast(enabled=self.use_amp):
                    model_kwargs = {'img': xt, 'timesteps': t.float()}
                    model_kwargs.update(condition)
                    pred_noise = self.model(**model_kwargs)
                    loss = F.mse_loss(pred_noise, noise)
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

            if (self.global_step + 1) % self.checkpoint_frequency == 0:
                self.save_checkpoint()

            self.global_step += 1
            if self.global_step >= self.num_train_steps:
                 break

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
