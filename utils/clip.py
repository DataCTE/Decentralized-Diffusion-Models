"""CLIP text encoder for Decentralized Diffusion Models."""

import torch
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class CLIPTextEncoder:
    """Text encoder using CLIP for text conditioning in diffusion models"""
    def __init__(self, device, config):
        self.device = device
        self.config = config
        
        # Handle model loading paths
        clip_model_path = config.clip_model
        if not os.path.exists(clip_model_path) and "/" not in clip_model_path:
            # Assume it's a HuggingFace model ID
            logger.info(f"Loading CLIP from HuggingFace: {clip_model_path}")
        else:
            logger.info(f"Loading CLIP from local path: {clip_model_path}")
        
        try:
            # Load CLIP text encoder from config
            self.text_encoder = CLIPTextModel.from_pretrained(
                clip_model_path,
                torch_dtype=torch.float16 if hasattr(config, 'use_mixed_precision') and config.use_mixed_precision else torch.float32
            ).to(device)
            
            # Load tokenizer - try both specialized CLIP tokenizer and AutoTokenizer
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
                logger.info("Loaded specialized CLIP tokenizer")
            except:
                logger.info("Falling back to AutoTokenizer for CLIP")
                self.tokenizer = AutoTokenizer.from_pretrained(clip_model_path)
            
            # Set to evaluation mode and freeze parameters
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
                
            logger.info(f"CLIP text encoder loaded successfully")
            
            # Store max token length
            self.max_length = getattr(config, 'max_token_length', 77)
            
        except Exception as e:
            logger.error(f"Error loading CLIP: {str(e)}")
            raise RuntimeError(f"Failed to load CLIP: {str(e)}")
            
    def encode(self, prompts, return_pooled=False):
        """
        Encode text prompts to embeddings
        
        Args:
            prompts: List of text prompts or single text prompt
            return_pooled: Whether to return pooled embeddings (for classifier-free guidance)
            
        Returns:
            Text embeddings tensor or tuple of (embeddings, pooled_embeddings)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        with torch.no_grad():
            # Tokenize and move to device
            text_inputs = self.tokenizer(
                prompts, 
                padding="max_length", 
                max_length=self.max_length, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get text embeddings from CLIP
            if return_pooled:
                outputs = self.text_encoder(**text_inputs, output_hidden_states=True)
                # Get both sequence embeddings and pooled embedding
                text_embeddings = outputs.last_hidden_state
                pooled_embeddings = outputs.pooler_output
                
                # Return embeddings in float16 format for consistency
                return (
                    text_embeddings.to(dtype=torch.float16 if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision else torch.float32),
                    pooled_embeddings.to(dtype=torch.float16 if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision else torch.float32)
                )
            else:
                text_embeddings = self.text_encoder(**text_inputs).last_hidden_state
                
                # Return embeddings in appropriate format for consistency
                return text_embeddings.to(dtype=torch.float16 if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision else torch.float32)
    
    def encode_with_uncond(self, prompts):
        """
        Encode text prompts along with empty prompt for classifier-free guidance
        
        Args:
            prompts: List of text prompts or single text prompt
            
        Returns:
            Tuple of (text_embeddings, uncond_embeddings)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Create empty prompts for unconditional guidance
        uncond_prompts = [""] * len(prompts)
        
        # Encode both text and unconditional prompts
        text_embeddings = self.encode(prompts)
        uncond_embeddings = self.encode(uncond_prompts)
        
        return text_embeddings, uncond_embeddings
        
    def apply_guidance(self, text_embeddings, uncond_embeddings, guidance_scale=7.5):
        """
        Apply classifier-free guidance scaling
        
        Args:
            text_embeddings: Conditional text embeddings
            uncond_embeddings: Unconditional text embeddings
            guidance_scale: Guidance scale (higher values = stronger adherence to prompt)
            
        Returns:
            Combined embeddings with guidance scaling
        """
        # Classic classifier-free guidance formula
        return uncond_embeddings + guidance_scale * (text_embeddings - uncond_embeddings) 