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
        self.config = config # Store the whole config
        # Use float32 by default, only use float16 if explicitly enabled AND cuda available
        self.precision = torch.float16 if getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available() else torch.float32
        self.clip_model_name_or_path = getattr(config, 'clip_model_name', 'openai/clip-vit-large-patch14') # Use configured name or default

        # Handle model loading paths
        if not os.path.exists(self.clip_model_name_or_path) and "/" in self.clip_model_name_or_path:
            # Assume it's a HuggingFace model ID if it contains '/' and doesn't exist locally
            logger.info(f"Loading CLIP from HuggingFace: {self.clip_model_name_or_path}")
        else:
            # Assume local path otherwise (even if it doesn't contain '/')
            logger.info(f"Loading CLIP from local path: {self.clip_model_name_or_path}")

        try:
            # Load CLIP text encoder
            # Load in float32 first, then convert if needed
            self.model = CLIPTextModel.from_pretrained(self.clip_model_name_or_path).to(self.device)

            # Load tokenizer - try both specialized CLIP tokenizer and AutoTokenizer
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name_or_path)
                logger.info("Loaded specialized CLIP tokenizer")
            except Exception:
                logger.info("Falling back to AutoTokenizer for CLIP")
                self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model_name_or_path)

            # Apply precision after loading and moving
            if self.precision == torch.float16:
                self.model.half()

            self.model.eval()
            # Freeze parameters
            for param in self.model.parameters():
                param.requires_grad_(False)

            logger.info(f"CLIP text encoder loaded successfully in {self.precision} precision.")

            # Store max token length from config or default
            self.max_length = getattr(config, 'clip_max_token_length', 77) # Use specific config name or default

        except Exception as e:
            logger.error(f"Error loading CLIP ({self.clip_model_name_or_path}): {str(e)}")
            raise RuntimeError(f"Failed to load CLIP: {str(e)}")

    @torch.no_grad()
    def encode_sequence(self, text):
        """Encodes text and returns the last_hidden_state (sequence embeddings)."""
        # Determine context manager based on precision
        context = torch.autocast(device_type=self.device.type, dtype=self.precision) if self.precision == torch.float16 else torch.no_grad()

        with context:
            inputs = self.tokenizer(
                 text,
                 return_tensors="pt",
                 padding="max_length", # Pad to max_length
                 truncation=True,
                 max_length=self.max_length
            )
            # Ensure input_ids are on the correct device
            input_ids = inputs.input_ids.to(self.device)
            # attention_mask might not be needed by CLIPTextModel if padding correctly handled? Verify.
            # attention_mask = inputs.attention_mask.to(self.device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=False, return_dict=True)
            # Return sequence embeddings in float32 as expected by some downstream models
            return outputs.last_hidden_state.to(torch.float32)

    @torch.no_grad()
    def encode_pooled(self, text):
        """Encodes text and returns the pooler_output (single vector per prompt)."""
        # Determine context manager based on precision
        context = torch.autocast(device_type=self.device.type, dtype=self.precision) if self.precision == torch.float16 else torch.no_grad()

        with context:
            inputs = self.tokenizer(
                 text,
                 return_tensors="pt",
                 padding="max_length", # Pad to max_length
                 truncation=True,
                 max_length=self.max_length
            )
            # Ensure input_ids are on the correct device
            input_ids = inputs.input_ids.to(self.device)
            # attention_mask = inputs.attention_mask.to(self.device) # May not be needed
            outputs = self.model(input_ids=input_ids, output_hidden_states=False, return_dict=True)

            if outputs.pooler_output is None:
                logger.warning("CLIP model did not return pooler_output. Returning zeros.")
                # Attempt to find the embedding dimension if possible
                embed_dim = self.model.config.hidden_size if hasattr(self.model, 'config') else 768
                batch_size = input_ids.shape[0] if isinstance(text, list) else 1
                return torch.zeros(batch_size, embed_dim, device=self.device, dtype=torch.float32)

            # Return pooled output in float32
            return outputs.pooler_output.to(torch.float32)

    # Keep encode as alias for pooled for backward compatibility if needed, or remove if confusing
    # def encode(self, text):
    #     return self.encode_pooled(text)

    def encode_with_uncond(self, prompts):
        """Encodes prompts and returns pooled outputs for conditional and unconditional."""
        if isinstance(prompts, str):
            prompts = [prompts]
        uncond_prompts = [""] * len(prompts)

        text_embeddings = self.encode_pooled(prompts)
        uncond_embeddings = self.encode_pooled(uncond_prompts)

        return text_embeddings, uncond_embeddings

    # apply_guidance usually works on noise predictions, not embeddings directly,
    # but keeping structure if it was used differently elsewhere.
    def apply_guidance(self, text_embeddings, uncond_embeddings, guidance_scale=7.5):
        return uncond_embeddings + guidance_scale * (text_embeddings - uncond_embeddings) 