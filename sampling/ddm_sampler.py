import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Dict, Union
from einops import rearrange # <-- Import rearrange

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
        patch_size: int, # <-- ADDED: Patch size used during expert training
        num_diffusion_timesteps: int = 1000,
        beta_start: float = 0.0001, # <-- ADDED: Schedule parameters
        beta_end: float = 0.02,     # <-- ADDED: Schedule parameters
    ):
        if not experts:
            raise ValueError("At least one expert model must be provided.")
        if router.num_clusters != len(experts):
             raise ValueError(f"Router num_clusters ({router.num_clusters}) must match the number of experts ({len(experts)}).")
        if patch_size <= 0:
             raise ValueError("patch_size must be a positive integer.") # <-- ADDED: Validation

        self.router = router.to(device).eval() # Ensure router is on device and in eval mode
        self.experts = [expert.to(device).eval() for expert in experts] # Ensure experts are on device and in eval mode
        self.num_experts = len(experts)
        self.device = device
        self.patch_size = patch_size # <-- STORED
        self.num_diffusion_timesteps = num_diffusion_timesteps

        # --- Noise Schedule ---
        # Use passed schedule parameters
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_linear_noise_schedule(
            timesteps=self.num_diffusion_timesteps,
            beta_start=beta_start, # <-- Use passed value
            beta_end=beta_end      # <-- Use passed value
        )
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
        # Need alphas for DDIM / alternative reverse steps if not using simple Euler
        self.alphas_cumprod = self.sqrt_alphas_cumprod ** 2
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        # Ensure variance calculation avoids division by zero for the last step if alpha_cumprod[T-1] is near 0
        self.variance = (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod.clamp(min=1e-20)) * \
                        (1.0 - self.alphas_cumprod / self.alphas_cumprod_prev.clamp(min=1e-20))
        self.variance = self.variance.clamp(min=1e-20) # Ensure variance is non-negative

        # Precompute schedule tensors for efficiency (optional, simple lookup is usually fast enough)
        # self.schedule_cache = {} # Removed cache for simplicity

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

        B, C, H, W = initial_noise.shape # Get original dimensions
        # --- ADDED: Check dimensions vs patch size ---
        if H % self.patch_size != 0 or W % self.patch_size != 0:
             raise ValueError(f"Input noise dimensions ({H}x{W}) not divisible by patch size ({self.patch_size}).")
        ph = pw = self.patch_size
        num_h_patches = H // ph
        num_w_patches = W // pw
        # -------------------------------------------

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

            # --- START EDIT: Patch xt before passing to experts ---
            xt_patched = rearrange(xt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)
            # --- END EDIT ---

            # 2. Get expert prediction(s) based on strategy
            # --- Modify expert_kwargs to use patched input ---
            expert_kwargs = {'img': xt_patched, 'timesteps': t_curr_vec}
            # ----------------------------------------------
            if expert_cond:
                 expert_kwargs.update(expert_cond) # Assumes ExpertModel forward accepts these keys

            pred_noise_patched = None # Initialize variable

            if strategy == 'top-1':
                probs = F.softmax(router_logits, dim=-1)
                top_expert_indices = torch.argmax(probs, dim=-1) # [B]

                # --- Loop remains, but input is patched ---
                pred_noise_patched = torch.zeros_like(xt_patched) # Predict in patched format
                for b_idx in range(B):
                     expert_idx = top_expert_indices[b_idx].item()
                     single_expert_kwargs = {}
                     for k, v in expert_kwargs.items():
                          if isinstance(v, torch.Tensor) and v.shape[0] == B:
                              single_expert_kwargs[k] = v[b_idx:b_idx+1]
                          else:
                              single_expert_kwargs[k] = v
                     # Expert predicts noise in patched format
                     pred_noise_patched[b_idx:b_idx+1] = self.experts[expert_idx](**single_expert_kwargs)
                 # ---------------------------------------

            elif strategy == 'full':
                probs = F.softmax(router_logits, dim=-1) # [B, num_experts]
                ensemble_pred_patched = torch.zeros_like(xt_patched) # Ensemble in patched format
                for k in range(self.num_experts):
                    # Expert predicts noise in patched format
                    expert_pred_patched = self.experts[k](**expert_kwargs)
                    weights = probs[:, k].view(B, *([1] * (xt_patched.ndim - 1))) # Adapt weights shape to patched data [B, 1, 1]
                    ensemble_pred_patched += weights * expert_pred_patched
                pred_noise_patched = ensemble_pred_patched # Use the combined patched prediction

            # --- START EDIT: Unpatch the predicted noise ---
            # Check if pred_noise_patched was actually computed
            if pred_noise_patched is None:
                 raise RuntimeError(f"Predicted noise was not computed for strategy '{strategy}'")

            # Rearrange back from [B, NumPatches, PatchDim] to [B, C, H, W]
            # PatchDim = c * ph * pw
            pred_noise = rearrange(pred_noise_patched, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                                   h=num_h_patches, w=num_w_patches, ph=ph, pw=pw, c=C)
            # --- END EDIT ---

            # --- Reverse Step ---
            # Using DDPM-like reverse step with the *unpatched* pred_noise
            params = self._get_schedule_params(t_curr_val.item(), t_prev_val.item())

            sqrt_alpha_curr = params['sqrt_alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            sqrt_one_minus_alpha_curr = params['sqrt_one_minus_alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            # Use the unpatched pred_noise here
            x0_pred = (xt - sqrt_one_minus_alpha_curr * pred_noise) / sqrt_alpha_curr.clamp(min=1e-8) # Added clamp for safety
            x0_pred.clamp_(-1.0, 1.0) # Optional clamping

            alpha_prev = params['alpha_prev'].view(-1, *([1] * (xt.ndim - 1)))
            alpha_curr = params['alpha_curr'].view(-1, *([1] * (xt.ndim - 1)))
            # Effective beta calculation might need clamp for stability if alpha_prev is 1.0 (first step)
            beta_curr = 1.0 - alpha_curr / alpha_prev.clamp(min=1e-8) # Added clamp
            beta_curr = beta_curr.clamp(min=0.0) # Ensure beta is non-negative

            mean_pred = ( (torch.sqrt(alpha_prev) * beta_curr / (1.0 - alpha_curr).clamp(min=1e-8)) * x0_pred +
                          (torch.sqrt((1.0 - alpha_prev).clamp(min=0.0)) * (1.0 - beta_curr) / (1.0 - alpha_curr).clamp(min=1e-8)) * xt )
                          # Modified: sqrt(1-alpha_prev) * (1-beta_curr) / (1-alpha_curr) <-- check DDPM formula
                          # Original DDPM: sqrt(alpha_prev)*(1-beta_curr)/(1-alpha_curr) * xt  <-- This seems more likely intended. Let's use standard:
            # Correct DDPM Mean Coefficient for xt: sqrt(alpha_prev)*(1-beta_curr)/(1-alpha_curr) -> simplifies to sqrt(alpha_curr_prev)*(1-beta)/ (1-alpha_curr)
            # Correct DDPM Mean Coefficient for x0: sqrt(alpha_prev)*beta/(1-alpha_curr)
            
            # Let's rewrite using standard DDPM terms for clarity:
            posterior_mean_coef1 = torch.sqrt(alpha_prev) * beta_curr / (1.0 - alpha_curr).clamp(min=1e-8)
            posterior_mean_coef2 = torch.sqrt(alpha_curr) * (1.0 - alpha_prev) / (1.0 - alpha_curr).clamp(min=1e-8) # Coef for xt according to DDPM paper Eq. 7
            # posterior_mean_coef2 simplified: sqrt(1-beta_curr) * (1-alpha_prev) / (1-alpha_curr) <-- seems correct from derived form
            
            mean_pred = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * xt

            variance = params['variance'].view(-1, *([1] * (xt.ndim - 1)))
            log_variance = torch.log(variance) # Variance is already clamped in __init__
            noise = torch.randn_like(xt) if t_prev_val > 0 else torch.zeros_like(xt)
            # Use eta=1 for DDPM variance, eta=0 for DDIM
            eta = 1.0 # For DDPM like variance
            xt = mean_pred + eta * (0.5 * log_variance).exp() * noise

        return xt # Return the final sample at t=0 