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
        self.precision = torch.float16 if config.use_mixed_precision else torch.float32
        
        # Handle model loading paths
        clip_model_path = config.clip_model
        if not os.path.exists(clip_model_path) and "/" not in clip_model_path:
            # Assume it's a HuggingFace model ID
            logger.info(f"Loading CLIP from HuggingFace: {clip_model_path}")
        else:
            logger.info(f"Loading CLIP from local path: {clip_model_path}")
        
        try:
            # Load CLIP text encoder from config
            self.model = CLIPTextModel.from_pretrained(clip_model_path).to(device)
            self.model.eval()
            
            if config.use_mixed_precision:
                self.model.half()
            
            # Load tokenizer - try both specialized CLIP tokenizer and AutoTokenizer
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
                logger.info("Loaded specialized CLIP tokenizer")
            except:
                logger.info("Falling back to AutoTokenizer for CLIP")
                self.tokenizer = AutoTokenizer.from_pretrained(clip_model_path)
            
            # Set to evaluation mode and freeze parameters
            for param in self.model.parameters():
                param.requires_grad_(False)
                
            logger.info(f"CLIP text encoder loaded successfully")
            
            # Store max token length
            self.max_length = getattr(config, 'max_token_length', 77)
            
        except Exception as e:
            logger.error(f"Error loading CLIP: {str(e)}")
            raise RuntimeError(f"Failed to load CLIP: {str(e)}")
            
    def encode(self, text):
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            return self.model(inputs.input_ids.to(self.device)).last_hidden_state.to(torch.float32)
    
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