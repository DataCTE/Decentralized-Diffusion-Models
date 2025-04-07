import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Dict, Union

# Assuming models and noise schedule are accessible
from models.router import RouterModel
from models.expert import ExpertModel
# Assuming noise schedule functions are available, e.g., from trainers or a util file
# Need functions to get alpha/sigma/etc. at specific timesteps 't'
# Example placeholder: get_schedule_params(t) -> dict
from trainers.trainer import get_linear_noise_schedule # Using placeholder from trainer for now

class DDMSampler:
    """
    Handles sampling from a Decentralized Diffusion Model ensemble.
    """
    def __init__(
        self,
        router: RouterModel,
        experts: List[ExpertModel],
        device: torch.device,
        num_diffusion_timesteps: int = 1000,
    ):
        if not experts:
            raise ValueError("At least one expert model must be provided.")
        if router.num_clusters != len(experts):
             raise ValueError(f"Router num_clusters ({router.num_clusters}) must match the number of experts ({len(experts)}).")
             
        self.router = router.to(device).eval() # Ensure router is on device and in eval mode
        self.experts = [expert.to(device).eval() for expert in experts] # Ensure experts are on device and in eval mode
        self.num_experts = len(experts)
        self.device = device
        self.num_diffusion_timesteps = num_diffusion_timesteps

        # --- Noise Schedule ---
        # Load or define the noise schedule used during training
        # Using the same placeholder schedule from trainer.py for consistency
        # In a real setup, load this from config or a dedicated schedule module
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_linear_noise_schedule(
            timesteps=self.num_diffusion_timesteps
        )
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        # Need alphas for DDIM / alternative reverse steps if not using simple Euler
        self.alphas_cumprod = self.sqrt_alphas_cumprod ** 2
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.variance = (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) * (1.0 - self.alphas_cumprod / self.alphas_cumprod_prev)

        # Precompute schedule tensors for efficiency
        self.schedule_cache = {}

    def _get_schedule_params(self, t_curr_val, t_prev_val):
         """ Helper to get schedule values for current and previous timesteps """
         # Convert float t (0-1) to integer timestep index if necessary
         # Assuming t_curr_val, t_prev_val are floats in [0, 1] mapped linearly
         t_curr_idx = int(t_curr_val * self.num_diffusion_timesteps)
         t_prev_idx = int(t_prev_val * self.num_diffusion_timesteps)
         # Clamp indices to valid range [0, num_diffusion_timesteps - 1]
         t_curr_idx = max(0, min(t_curr_idx, self.num_diffusion_timesteps - 1))
         t_prev_idx = max(0, min(t_prev_idx, self.num_diffusion_timesteps - 1))
         
         # Retrieve precomputed values using integer indices
         # Example: retrieving values needed for DDPM reverse step (may vary based on sampler)
         params = {
             'sqrt_alpha_curr': self.sqrt_alphas_cumprod[t_curr_idx],
             'sqrt_one_minus_alpha_curr': self.sqrt_one_minus_alphas_cumprod[t_curr_idx],
             'sqrt_alpha_prev': self.sqrt_alphas_cumprod[t_prev_idx],
             'sqrt_one_minus_alpha_prev': self.sqrt_one_minus_alphas_cumprod[t_prev_idx],
             'alpha_curr': self.alphas_cumprod[t_curr_idx],
             'alpha_prev': self.alphas_cumprod[t_prev_idx],
             'variance': self.variance[t_curr_idx]
         }
         return params


    @torch.no_grad()
    def sample(
        self,
        initial_noise: torch.Tensor, # Typically shape [B, C, H, W] for image latents
        num_steps: int, # Number of inference steps (can be different from training T)
        conditioning: Optional[Dict[str, torch.Tensor]] = None, # Dict mapping condition names to tensors
        strategy: str = 'top-1', # 'top-1' or 'full'
        show_progress: bool = True,
    ):
        """
        Generates samples using the DDM ensemble.

        Args:
            initial_noise (torch.Tensor): Starting noise tensor (usually N(0,I) at t=1).
            num_steps (int): Number of reverse diffusion steps to perform.
            conditioning (Optional[Dict[str, torch.Tensor]]): Dictionary containing conditioning
                tensors required by the router and experts (e.g., 'y' for CLIP embedding).
                Keys should match expected arguments.
            strategy (str): Inference strategy ('top-1' or 'full'). Defaults to 'top-1'.
            show_progress (bool): Whether to display a progress bar.

        Returns:
            torch.Tensor: The generated sample tensor (typically at t=0).
        """
        if strategy not in ['top-1', 'full']:
            raise ValueError("Strategy must be 'top-1' or 'full'")

        B = initial_noise.shape[0]
        xt = initial_noise.to(self.device)
        
        # Prepare conditioning for router and experts
        router_cond = None
        expert_cond = {}
        if conditioning:
            # Router typically needs 'y' if conditioned
            if self.router.has_cond:
                 router_cond = conditioning.get('y') # Assuming 'y' is the key for router condition
                 if router_cond is None:
                      raise ValueError("Router expects conditioning 'y', but it was not found in the conditioning dict.")
                 router_cond = router_cond.to(self.device)
            # Experts need all their specific conditions
            # Pass the whole dictionary, ExpertModel forward should handle it
            expert_cond = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in conditioning.items()}


        # Define the timestep schedule for inference
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device) # t=1 to t=0

        iterator = tqdm(range(num_steps), desc="DDM Sampling", disable=not show_progress)
        for i in iterator:
            t_curr_val = timesteps[i]
            t_prev_val = timesteps[i+1]

            t_curr_vec = torch.full((B,), t_curr_val * self.num_diffusion_timesteps, device=self.device).float() # Scale t for model input

            # --- Get Prediction ---
            # 1. Get router prediction (logits)
            router_kwargs = {'x': xt, 't': t_curr_vec}
            if router_cond is not None:
                router_kwargs['y'] = router_cond
            router_logits = self.router(**router_kwargs) # [B, num_experts]

            # 2. Get expert prediction(s) based on strategy
            expert_kwargs = {'img': xt, 'timesteps': t_curr_vec}
            # Merge expert-specific conditioning into kwargs
            if expert_cond:
                 expert_kwargs.update(expert_cond) # Assumes ExpertModel forward accepts these keys

            if strategy == 'top-1':
                # Find the index of the expert with the highest probability
                probs = F.softmax(router_logits, dim=-1)
                top_expert_indices = torch.argmax(probs, dim=-1) # [B]

                # Get predictions only from the chosen expert for each batch item
                # This requires careful batching or a loop if experts can't handle sparse batch indices easily
                # Simple loop approach (less efficient but clear):
                pred_noise = torch.zeros_like(xt)
                for b_idx in range(B):
                     expert_idx = top_expert_indices[b_idx].item()
                     # Prepare single-item input for the expert
                     single_expert_kwargs = {}
                     for k, v in expert_kwargs.items():
                          if isinstance(v, torch.Tensor) and v.shape[0] == B:
                              single_expert_kwargs[k] = v[b_idx:b_idx+1]
                          else: # Keep non-batch tensors as is (like img_ids if constant)
                              single_expert_kwargs[k] = v
                              
                     pred_noise[b_idx:b_idx+1] = self.experts[expert_idx](**single_expert_kwargs)
                # Batched gather is complex; loop retained as functional implementation.

            elif strategy == 'full':
                # Weighted average of all expert predictions
                probs = F.softmax(router_logits, dim=-1) # [B, num_experts]
                ensemble_pred = torch.zeros_like(xt)
                for k in range(self.num_experts):
                    expert_pred = self.experts[k](**expert_kwargs) # [B, C, H, W] or similar
                    # Get weights for expert k for each batch item: probs[:, k] -> [B]
                    # Reshape weights for broadcasting: [B, 1, 1, ...]
                    weights = probs[:, k].view(B, *([1] * (xt.ndim - 1)))
                    ensemble_pred += weights * expert_pred
                pred_noise = ensemble_pred # Use the combined prediction

            # --- Reverse Step ---
            # Using DDPM-like reverse step, assuming model predicts noise epsilon
            # Get schedule parameters for current and previous timesteps
            params = self._get_schedule_params(t_curr_val.item(), t_prev_val.item())

            # Predict x0 based on the noise prediction
            sqrt_alpha_curr = params['sqrt_alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            sqrt_one_minus_alpha_curr = params['sqrt_one_minus_alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            x0_pred = (xt - sqrt_one_minus_alpha_curr * pred_noise) / sqrt_alpha_curr
            x0_pred.clamp_(-1.0, 1.0) # Optional clamping based on data range

            # Calculate mean of q(x_{t-1} | xt, x0)
            alpha_prev = params['alpha_prev'].view(-1, *([1] * (xt.ndim - 1)))
            alpha_curr = params['alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            beta_curr = 1.0 - alpha_curr / alpha_prev # Calculate beta effectively
            
            mean_pred = ( (torch.sqrt(alpha_prev) * beta_curr / (1.0 - alpha_curr)) * x0_pred +
                          (torch.sqrt(1.0 - beta_curr) * (1.0 - alpha_prev) / (1.0 - alpha_curr)) * xt )

            # Add variance (noise term) - optional for DDIM (eta=0)
            variance = params['variance'].view(-1, *([1] * (xt.ndim - 1)))
            log_variance = torch.log(variance.clamp(min=1e-20))
            noise = torch.randn_like(xt) if t_prev_val > 0 else torch.zeros_like(xt)
            xt = mean_pred + (0.5 * log_variance).exp() * noise


            # --- Alternative: Simple Euler Step ---
            # Assumes 'pred_noise' is actually velocity 'v' or scaled score
            # step_size = t_curr_val - t_prev_val # dt is negative here
            # xt = xt + step_size * pred_noise # This is simplified Euler step v(x_t, t) dt
            # ------------------------------------

        return xt # Return the final sample at t=0 