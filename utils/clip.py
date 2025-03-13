"""CLIP text encoder for Decentralized Diffusion Models."""

import torch
from transformers import CLIPTextModel, AutoTokenizer

class CLIPTextEncoder:
    """Text encoder using CLIP for text conditioning in diffusion models"""
    def __init__(self, device, config):
        self.device = device
        
        # Load CLIP text encoder from config
        self.text_encoder = CLIPTextModel.from_pretrained(
            config.clip_model,
            torch_dtype=torch.float16  # Use half precision for efficiency
        ).to(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.clip_model
        )
        
        # Set to evaluation mode and freeze parameters
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
            
    def encode(self, prompts):
        """Encode text prompts to embeddings"""
        with torch.no_grad():
            # Tokenize and move to device
            text_inputs = self.tokenizer(
                prompts, 
                padding="max_length", 
                max_length=77, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get text embeddings from CLIP
            text_embeddings = self.text_encoder(**text_inputs).last_hidden_state
            
            # Return embeddings in float16 format for consistency
            return text_embeddings.to(dtype=torch.float16) 