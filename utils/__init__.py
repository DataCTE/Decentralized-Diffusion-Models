"""Utility functions for Decentralized Diffusion Models.""" 

from .fsdp import *
import os # Add os import if needed for find_latest_checkpoint
import glob # Add glob import if needed for find_latest_checkpoint
from types import SimpleNamespace # Add SimpleNamespace import


# Helper to recursively convert dict to SimpleNamespace
def dict_to_sns(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_sns(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_sns(item) for item in d]
    return d

# Add find_latest_checkpoint here as well if it's not elsewhere
def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file in a directory based on step number or modification time."""
    checkpoint_files = list(glob.glob(os.path.join(checkpoint_dir, "*.pt"))) \
                     + list(glob.glob(os.path.join(checkpoint_dir, "*.pth"))) # Include .pth too
    
    if not checkpoint_files:
        return None

    # Prioritize checkpoints with step numbers in the filename (e.g., step_10000.pt)
    step_checkpoints = {}
    for f in checkpoint_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) > 1 and parts[-1].replace('.pt', '').replace('.pth', '').isdigit():
             try:
                  step = int(parts[-1].replace('.pt', '').replace('.pth', ''))
                  step_checkpoints[step] = f
             except ValueError:
                  continue # Not a step number

    if step_checkpoints:
        latest_step = max(step_checkpoints.keys())
        return step_checkpoints[latest_step]
    else:
        # Fallback to modification time if no step numbers found
        return max(checkpoint_files, key=os.path.getmtime)

# --- Add any other shared utility functions here --- 