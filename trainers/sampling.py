"""Sampling utilities for Decentralized Diffusion Models."""

import torch
import logging
from tqdm import tqdm
from trainers.diffusion import (
    get_alphas_and_betas, 
    ddim_step
)
import torch.nn.functional as F
from collections import defaultdict
from config import get_config
import torch.distributed as dist
from config import get_config
from einops import rearrange
from typing import Callable, Dict, Tuple
import torch.nn as nn

logger = logging.getLogger(__name__)

# test add
def ddm_sample(
    router,
    experts,
    shape,
    num_steps=50,
    device=None,
    cfg_scale=7.5,
    text_embeddings=None,
    uncond_embeddings=None,
    eta=0.0,
    verbose=True,
    temperature=1.0,
    inference_strategy="top_k",  # Add strategy parameter from paper
    top_k=1,                     # For top-k selection
    top_p=0.9,                   # For nucleus sampling
    true_clusters=None,          # For oracle evaluation
    config=None  # Add config parameter
):
    """DDM sampling with multiple inference strategies"""
    if device is None:
        device = next(router.parameters()).device
    
    # Get critical parameters from config
    patch_size = config.patch_size
    vec_in_dim = config.vec_in_dim
    
    # Initialize latents with proper dimensions
    latents = torch.randn(shape, device=device)
    
    # Create progress bar
    progress_bar = tqdm(range(num_steps), disable=not verbose)
    
    for step in progress_bar:
        # Get router predictions with ORIGINAL latents
        router_logits = router(
            img=latents,  # Use original 4D latents
            timesteps=torch.full((latents.shape[0],), step, device=device),
            txt=text_embeddings
        )
        
        # Prepare expert input with proper patching
        img_input = rearrange(
            latents,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=config.patch_size,
            p2=config.patch_size
        )
        
        # Generate position IDs based on actual spatial dimensions
        H_patch = latents.shape[2] // patch_size
        W_patch = latents.shape[3] // patch_size
        pos_h = torch.arange(H_patch, device=device)
        pos_w = torch.arange(W_patch, device=device)
        img_ids = torch.stack(torch.meshgrid(pos_h, pos_w, indexing='ij'), -1)
        img_ids = img_ids.reshape(1, -1, 2).expand(latents.shape[0], -1, -1)

        router_weights = F.softmax(router_logits / temperature, dim=-1)
        
        # Paper's inference strategies (Section 3.5)
        if inference_strategy == "full":
            # Full ensemble - use all experts
            selected_weights = router_weights
            selected_indices = torch.arange(len(experts), device=device).expand(latents.shape[0], -1)
        elif inference_strategy == "top_k":
            # Greedy top-k expert selection
            selected_weights, selected_indices = router_weights.topk(top_k, dim=-1)
        elif inference_strategy == "sample":
            # Stochastic sampling
            if torch.isnan(router_weights).any() or torch.isinf(router_weights).any() or (router_weights < 0).any():
                #print("Warning: Invalid values detected in router_weights before multinomial sampling!")
                #print("router_weights min:", router_weights.min())
                #print("router_weights max:", router_weights.max())
                router_weights = torch.clamp(router_weights, min=0, max=1) # Clamp to valid probability range

            # Ensure router_weights are valid probabilities for multinomial sampling
            router_weights = torch.where(torch.isnan(router_weights) | torch.isinf(router_weights) | (router_weights < 0), torch.zeros_like(router_weights), router_weights)
            router_weights = torch.clamp(router_weights, min=0, max=1) # Double clamp to be safe

            if inference_strategy == "sample":
                #print("Inference strategy is sample")
                #print("Sum of router_weights:", router_weights.sum())
                #print("Min of router_weights:", router_weights.min())
                #print("Max of router_weights:", router_weights.max())
                #print("Shape of router_weights:", router_weights.shape)
                #print("Sample values of router_weights:", router_weights[:2])

                selected_indices = torch.multinomial(router_weights, 1).squeeze(-1)
                #print("Shape of selected_indices:", selected_indices.shape)
                #print("Type of selected_indices:", selected_indices.dtype)
                #print("Min value of selected_indices:", selected_indices.min())
                #print("Max value of selected_indices:", selected_indices.max())
                #print("Sample values of selected_indices:", selected_indices[:10])

            selected_weights = torch.ones_like(selected_indices, dtype=torch.float32)
        elif inference_strategy == "nucleus":
            # Nucleus sampling (top-p)
            sorted_weights, sorted_indices = torch.sort(router_weights, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_weights, dim=-1)
            mask = cum_probs <= top_p
            mask[..., 0] = True  # Ensure at least one expert
            selected_weights = sorted_weights[mask]
            selected_indices = sorted_indices[mask]
        elif inference_strategy == "oracle" and true_clusters is not None:
            # Ground truth expert usage (for validation)
            selected_indices = true_clusters
            selected_weights = torch.ones_like(selected_indices, dtype=torch.float32)
        elif inference_strategy == "stochastic":
            # Stochastic expert sampling
            selected_indices = torch.multinomial(router_weights, 1).squeeze(-1)
        else:
            raise ValueError(f"Invalid inference strategy: {inference_strategy}")

        # Convert tensor to numpy array and flatten
        selected_indices_np = selected_indices.cpu().numpy()
        flattened_indices = selected_indices_np.flatten().tolist()

        # Paper's batched expert execution
        expert_outputs = []
        for expert_idx in set(flattened_indices):
            # Create device-aware boolean mask
            mask = torch.tensor(
                [idx == expert_idx for idx in flattened_indices],
                device=device,
                dtype=torch.bool
            )
            
            # Get valid batch indices
            batch_indices = torch.nonzero(mask).squeeze()
            
            # Handle single sample case
            if batch_indices.dim() == 0:
                batch_indices = batch_indices.unsqueeze(0)
            
            # Extract relevant inputs
            expert = experts[expert_idx]
            expert_input = img_input[batch_indices]

            # Get actual dimensions from original latents
            H_patch = latents.shape[2] // config.patch_size
            W_patch = latents.shape[3] // config.patch_size
            
            # Reshape latent tensor to sequence format [B, L, C]
            B, L, C = expert_input.shape  # Correct dimension unpacking

            expert_timesteps = torch.full((B,), step, device=device)
            expert_text = text_embeddings[batch_indices] if text_embeddings is not None else None
            
            # Generate position IDs using calculated spatial dimensions
            img_ids = torch.stack(
                torch.meshgrid(
                    torch.arange(H_patch, device=device),
                    torch.arange(W_patch, device=device),
                    indexing='ij'
                ), -1
            ).reshape(1, -1, 2).expand(B, -1, -1)
            
            # Generate text position IDs
            txt_len = expert_text.shape[1] if expert_text is not None else 0
            txt_ids = torch.arange(txt_len, device=device).unsqueeze(0).repeat(expert_input.shape[0], 1)
            
            # Create 2D position IDs for text (matching image position dimensions)
            txt_ids = txt_ids[:, :, None].repeat(1, 1, 2)  # Shape: [B, txt_len, 2]

            # Create conditioning vector (y) from text embeddings
            y = expert_text.mean(dim=1) if expert_text is not None else torch.zeros(expert_input.shape[0], 
                                                                                   vec_in_dim, 
                                                                                   device=device)
            
            # Get cluster IDs from router selection
            cluster_ids = torch.full((expert_input.shape[0],), expert_idx, 
                                   device=device, dtype=torch.long)

            # Modified expert call with reshaping
            expert_output = expert(
                img=expert_input,
                img_ids=img_ids,
                txt=expert_text,
                txt_ids=txt_ids,
                timesteps=expert_timesteps,
                y=y,
                cluster_ids=cluster_ids
            )
            
            # Remove unnecessary reshape since output is already 4D
            reshaped_output = expert_output  # Directly use the model output
            
            expert_outputs.append(reshaped_output)
        
        # Combine predictions
        pred = torch.zeros_like(latents)
        output_idx = 0
        for idx in set(flattened_indices):
            mask = torch.tensor(
                [i == idx for i in flattened_indices],
                device=device,
                dtype=torch.bool
            )
            batch_indices = torch.nonzero(mask).squeeze()
            pred[batch_indices] = expert_outputs[output_idx]
            output_idx += 1

        #print("Shape of combined_pred before ddim_step:", pred.shape)
        #print("Shape of timestep before ddim_step:", step.shape)

        # DDIM update step with device-aware tensors
        latents = ddim_step(
            lambda x_t, t, c: pred,
            latents,
            torch.full((latents.shape[0],), step, device=device),  # Already on device
            torch.full((latents.shape[0],), step+1, device=device) if step < num_steps-1 else None,
            get_alphas_and_betas(num_steps, "cosine")[0].to(device),  # Add device transfer
            get_alphas_and_betas(num_steps, "cosine")[1].to(device),  # Add device transfer
            eta=eta
        )

    return latents

def distilled_sample(distilled_model, shape, num_steps=50, prompt_embeds=None, 
                    cfg_scale=7.5, device=None, eta=0.0, 
                    noise=None, callback=None):
    """
    Sampling using a distilled model (paper Section 3.6)
    
    Args:
        distilled_model: Distilled diffusion model
        shape: Output shape
        num_steps: Number of sampling steps
        prompt_embeds: Text prompt embeddings (optional)
        cfg_scale: Classifier-free guidance scale
        device: Device to use
        eta: Controls the amount of noise added
        noise: Initial noise (optional)
        callback: Callback function for intermediate steps (optional)
    """
    # Default device
    if device is None:
        device = next(distilled_model.parameters()).device
    
    # Create initial noise if not provided
    if noise is None:
        x = torch.randn(shape, device=device)
    else:
        x = noise.to(device)
    
    batch_size = shape[0]
    
    # Setup noise schedule
    alphas, alpha_bar, betas = get_alphas_and_betas(num_steps)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)
    
    # Progress bar
    progress = tqdm(range(num_steps), desc="Distilled Sampling")
    
    # Sampling loop
    for t in progress:
        # Current timestep
        batch_size = x.shape[0]
        timestep = torch.full((batch_size,), t, device=device)
        
        with torch.no_grad():
            # Get model prediction with conditional guidance if provided
            if prompt_embeds is not None and cfg_scale > 1.0:
                # Run both conditional and unconditional forward passes
                unconditional_latents = x.clone()
                conditional_latents = x.clone()
                
                # Get unconditional prediction
                uncond_pred = distilled_model(unconditional_latents, timestep)
                
                # Get conditional prediction
                cond_pred = distilled_model(conditional_latents, timestep, prompt_embeds)
                
                # Apply classifier-free guidance
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            else:
                # Simple forward pass without guidance
                pred = distilled_model(x, timestep)
            
            # DDIM step
            next_timestep = torch.full((batch_size,), t+1, device=device) if t < num_steps-1 else None
            
            if next_timestep is not None:
                # Update sample using DDIM step
                x = ddim_step(
                    lambda x_t, t, c: pred,  # Use precomputed prediction
                    x, 
                    timestep,
                    next_timestep,
                    alphas,
                    alpha_bar,
                    eta=eta
                )
            
            # Optional callback for visualization
            if callback and t % 5 == 0:
                callback(x, t)
    
    return x

def quantize_model_for_inference(model, dtype=torch.float16):
    """
    Quantize a model for memory-efficient inference
    
    Args:
        model: The model to quantize
        dtype: Target dtype for quantization (default: float16)
        
    Returns:
        Quantized model
    """
    # Simple quantization by converting to half precision
    model = model.to(dtype)
    logger.info(f"Converted model to {dtype} for efficient inference")
    return model

def ddm_sample_optimized(router, experts, shape, **kwargs):
    """
    Memory-optimized version that quantizes models
    """
    # Quantize models
    quantized_router = quantize_model_for_inference(router)
    quantized_experts = {
        idx: quantize_model_for_inference(expert)
        for idx, expert in experts.items()
    }
    
    return ddm_sample(
        router=quantized_router,
        experts=quantized_experts,
        shape=shape,
        **kwargs
    ) 