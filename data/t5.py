"""T5 text encoder for Decentralized Diffusion Models."""

import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class T5TextEncoder:
    """Text encoder using T5 for text conditioning in diffusion models"""
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.precision = torch.float16 if config.use_mixed_precision else torch.float32
        
        # Handle model loading paths
        t5_model_path = config.t5_model
        if not os.path.exists(t5_model_path) and "/" not in t5_model_path:
            # Assume it's a HuggingFace model ID
            logger.info(f"Loading T5 from HuggingFace: {t5_model_path}")
        
        try:
            # Load T5 text encoder from config
            self.model = T5EncoderModel.from_pretrained(t5_model_path).to(device)
            self.model.eval()
            
            if config.use_mixed_precision:
                self.model.half()
            
            # Load tokenizer - try specialized T5 tokenizer and AutoTokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
                logger.info("Loaded specialized T5 tokenizer")
            except:
                logger.info("Falling back to AutoTokenizer for T5")
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
            
            # Set to evaluation mode and freeze parameters
            for param in self.model.parameters():
                param.requires_grad_(False)
                
            logger.info(f"T5 text encoder loaded successfully")
            
            # Store max token length
            self.max_length = getattr(config, 'max_token_length', 128)
            
        except Exception as e:
            logger.error(f"Error loading T5: {str(e)}")
            raise RuntimeError(f"Failed to load T5: {str(e)}")
            
    def encode(self, text):
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            return self.model(inputs.input_ids.to(self.device)).last_hidden_state.to(torch.float32) 