from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer
import torch

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    """Print warning about missing or unexpected keys when loading model weights"""
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

class CLIPEmbedder(nn.Module):
    """Text embedder using CLIP only, optimized for diffusion conditioning"""
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.max_length = max_length
        
        # Initialize CLIP tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
        self.model = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        
        # Set to eval mode and freeze parameters
        self.model = self.model.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        """Encode text to embeddings using CLIP"""
        # Tokenize inputs
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # Get model output (no gradient tracking needed)
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(self.model.device),
                attention_mask=batch_encoding.get("attention_mask", None),
                output_hidden_states=False,
            )
            
        # Return the last hidden state (sequence of token embeddings)
        return outputs.last_hidden_state
    
    def encode_with_uncond(self, text: list[str]) -> tuple[Tensor, Tensor]:
        """Encode both text and empty string for classifier-free guidance"""
        text_embeds = self.forward(text)
        
        # Create batch of empty strings with same batch size
        uncond_text = [""] * len(text)
        uncond_embeds = self.forward(uncond_text)
        
        return text_embeds, uncond_embeds
