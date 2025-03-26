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
            # Load T5 text encoder from config with optimizations
            from transformers import AutoConfig
            t5_config = AutoConfig.from_pretrained(t5_model_path)
            
            # Enable performance optimizations
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            
            # Use optimized model loading for inference
            self.model = T5EncoderModel.from_pretrained(
                t5_model_path,
                config=t5_config,
                device_map=device,  # Use device mapping for efficient loading
                torch_dtype=self.precision,  # Use desired precision
            )
            
            self.model.eval()
            
            # Load tokenizer - try specialized T5 tokenizer and AutoTokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
                self.tokenizer.model_max_length = getattr(config, 'max_token_length', 128)
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
            
            # Pre-allocate attention mask for reuse
            self.cached_masks = {}
            
        except Exception as e:
            logger.error(f"Error loading T5: {str(e)}")
            raise RuntimeError(f"Failed to load T5: {str(e)}")
    
    def encode(self, text):
        """Encode text with performance optimizations"""
        # Use cached attention masks for common lengths to avoid recomputation
        batch_size = len(text)
        
        with torch.autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
            # Tokenize inputs
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding="longest", 
                truncation=True,
                max_length=self.max_length
            )
            
            # Get sequence length for this batch
            seq_len = inputs.input_ids.size(1)
            
            # Use cached attention mask if available for this sequence length
            if seq_len in self.cached_masks and self.cached_masks[seq_len].size(0) >= batch_size:
                # Reuse cached mask to avoid repetitive computations
                attention_mask = self.cached_masks[seq_len][:batch_size, :seq_len].clone()
            else:
                # Create and cache a new mask for this sequence length
                attention_mask = inputs.attention_mask.to(self.device)
                self.cached_masks[seq_len] = attention_mask.clone()
            
            # Perform inference with optimizations
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs.input_ids.to(self.device),
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True
                )
                
            return outputs.last_hidden_state.to(torch.float32) 