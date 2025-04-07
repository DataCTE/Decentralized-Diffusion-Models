import torch
import os
import glob
import numpy as np
from PIL import Image
from typing import Optional

def load_model_checkpoint(model: torch.nn.Module, filepath: str, device: torch.device):
    """Loads state dict from a checkpoint, handling potential 'module.' prefix."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    try:
        checkpoint = torch.load(filepath, map_location=device)
        # Determine the actual state dict
        if isinstance(checkpoint, dict):
            # Common keys: model_state_dict (from trainer), state_dict (direct save), model (less common)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
            # If after checking common keys, it's still a dict but not the state_dict itself,
            # maybe the checkpoint *is* the state_dict (e.g., just model weights saved)
            if not isinstance(state_dict, dict):
                 state_dict = checkpoint
        elif isinstance(checkpoint, torch.nn.Module): # If the entire model was saved
            state_dict = checkpoint.state_dict()
        else:
             raise TypeError(f"Unsupported checkpoint format at {filepath}. Expected dict or nn.Module.")

        if not state_dict:
             raise ValueError(f"Could not extract state_dict from checkpoint at {filepath}")

        # Handle 'module.' prefix (often added by DataParallel or DistributedDataParallel)
        adjusted_state_dict = {}
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        for k, v in state_dict.items():
            name = k[len("module."):] if has_module_prefix and k.startswith("module.") else k
            adjusted_state_dict[name] = v

        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)
        
        model_name = model.__class__.__name__
        if not missing_keys and not unexpected_keys:
             print(f"Successfully loaded weights strictly for {model_name} from {filepath}")
        else:
             if missing_keys: print(f"Warning: Missing keys when loading {model_name}: {missing_keys}")
             if unexpected_keys: print(f"Warning: Unexpected keys when loading {model_name}: {unexpected_keys}")
             print(f"Successfully loaded weights non-strictly for {model_name} from {filepath}")

    except Exception as e:
        print(f"Error loading checkpoint for {model.__class__.__name__} from {filepath}: {e}")
        raise e

def tensor_to_pil(tensor):
    """Converts a B C H W tensor in range [-1, 1] to a list of PIL Images."""
    # Ensure tensor is on CPU and denormalized
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0 # Denormalize from [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    # Convert to HWC uint8 format
    images_np = (tensor.permute(0, 2, 3, 1) * 255).numpy().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]
    return pil_images

def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> Optional[str]:
    """
    Finds the latest checkpoint file in a directory based on modification time.

    Args:
        checkpoint_dir (str): The directory containing checkpoints.
        pattern (str): Glob pattern to match checkpoint files (e.g., "*.pt", "expert_*.pt").

    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if no matching files found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    try:
        list_of_files = glob.glob(os.path.join(checkpoint_dir, pattern))
        if not list_of_files:
            return None
        # Find the file with the latest modification time
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest checkpoint in {checkpoint_dir}: {e}")
        return None 