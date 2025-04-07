"""T5 text encoder for Decentralized Diffusion Models."""

import torch
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, AutoConfig
from transformers.utils import logging as hf_logging
import logging
import os

logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error() # Suppress HuggingFace warnings unless error

class T5TextEncoder:
    """Text encoder using T5 for text conditioning in diffusion models"""
    def __init__(self, device, config):
        self.device = device
        self.config = config # Store the whole config
        # Use float32 by default, only use float16 if explicitly enabled AND cuda available
        self.precision = torch.float16 if getattr(config, 'use_mixed_precision', False) and torch.cuda.is_available() else torch.float32
        self.t5_model_name_or_path = getattr(config, 't5_model_name', 'google/flan-t5-xl') # Use configured name or default

        # Handle model loading paths
        if not os.path.exists(self.t5_model_name_or_path) and "/" in self.t5_model_name_or_path:
            logger.info(f"Loading T5 from HuggingFace: {self.t5_model_name_or_path}")
        else:
            logger.info(f"Loading T5 from local path: {self.t5_model_name_or_path}")

        try:
            # Load T5 config first
            t5_config = AutoConfig.from_pretrained(self.t5_model_name_or_path)

            # Use device_map='auto' for better memory management or specify device
            # Note: device_map might conflict with manual .to(device) later if not careful
            load_device = 'auto' if torch.cuda.device_count() > 1 else self.device
            # Load in float32, then potentially cast
            self.model = T5EncoderModel.from_pretrained(
                self.t5_model_name_or_path,
                config=t5_config,
                device_map=load_device,
                # torch_dtype=self.precision, # Apply precision later
                low_cpu_mem_usage=True, # Useful for large models
            )

            # Move to target device if device_map didn't handle it fully
            if load_device != self.device and not isinstance(load_device, dict):
                 self.model.to(self.device)

            # Apply precision
            if self.precision == torch.float16:
                self.model.half()

            self.model.eval()

            # Load tokenizer - try specialized T5 tokenizer and AutoTokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name_or_path)
                logger.info("Loaded specialized T5 tokenizer")
            except Exception:
                logger.info("Falling back to AutoTokenizer for T5")
                self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name_or_path)

            # Set to evaluation mode and freeze parameters
            for param in self.model.parameters():
                param.requires_grad_(False)

            logger.info(f"T5 text encoder loaded successfully in {self.precision} precision.")

            # Store max token length from config or default
            self.max_length = getattr(config, 't5_max_token_length', 128) # Use specific config name or default
            # Set tokenizer max length if possible
            if hasattr(self.tokenizer, 'model_max_length'):
                 self.tokenizer.model_max_length = self.max_length


        except Exception as e:
            logger.error(f"Error loading T5 ({self.t5_model_name_or_path}): {str(e)}")
            raise RuntimeError(f"Failed to load T5: {str(e)}")

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