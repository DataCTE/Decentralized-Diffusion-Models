"""T5 text encoder for Decentralized Diffusion Models."""

import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, AutoConfig
from transformers.utils import logging as hf_logging
import logging
import os
import torch.nn as nn # Added for type hinting
from types import SimpleNamespace # Added for type hinting

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error() # Suppress HuggingFace warnings unless error

class T5TextEncoder(nn.Module):
    """Text encoder using T5 for text conditioning in diffusion models"""
    def __init__(self, device: torch.device, config: SimpleNamespace):
        super().__init__()
        self.device = device
        self.config = config # Store the whole config
        self.model_name = config.t5_model_name
        self.max_length = config.t5_max_token_length
        self.precision = torch.float16 if config.use_mixed_precision else torch.float32

        logger.info(f"Loading T5 from HuggingFace: {self.model_name}")
        try:
            # Explicitly set device_map to the target device
            self.model = T5EncoderModel.from_pretrained(
                self.model_name,
                torch_dtype=self.precision,
                low_cpu_mem_usage=True, # Recommended for large models
                device_map=self.device # <--- Force model parts to this device
            )
            self.model.eval() # Set to evaluation mode
            for param in self.model.parameters():
                param.requires_grad = False # Freeze parameters

            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                logger.info("Falling back to AutoTokenizer for T5")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Manually ensure model is on the correct device AGAIN after loading,
            # although device_map should handle it. This is belt-and-suspenders.
            self.model.to(self.device)

            logger.info(f"T5 text encoder loaded successfully in {self.precision} precision.")

            # Set tokenizer max length if possible
            if hasattr(self.tokenizer, 'model_max_length'):
                 self.tokenizer.model_max_length = self.max_length

        except Exception as e:
            logger.error(f"Failed to load T5 model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to load T5: {e}")

    @torch.no_grad()
    def encode(self, text):
        """Encode text and return last_hidden_state (sequence embeddings)."""
        if isinstance(text, str): text = [text] # Ensure list input
        batch_size = len(text)

        # Determine context manager based on precision
        context = torch.autocast(device_type=self.device.type, dtype=self.precision) if self.precision == torch.float16 else torch.no_grad()

        try:
            with context:
                # Tokenize inputs
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )

                # Ensure tensors are on the model's primary device (handles device_map)
                model_device = next(self.model.parameters()).device
                input_ids = inputs.input_ids.to(model_device)
                attention_mask = inputs.attention_mask.to(model_device)

                # Perform inference
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False, # We only need the last hidden state
                    return_dict=True
                )

                embeddings = outputs.last_hidden_state

                # Return embeddings on the main device, cast to float32
                return embeddings.to(self.device, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error during T5 encoding: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a zero tensor as fallback (with expected dimensions)
            # Try to get hidden size from config
            hidden_size = getattr(self.model.config, 'd_model', 1024) # Common T5 hidden size name
            return torch.zeros((batch_size, self.max_length, hidden_size), device=self.device, dtype=torch.float32) 