import torch
import torch.nn as nn
# Ensure the import path correctly points to the MMDiT implementation within your flux structure
from .flux.model import Flux as MMDiT # Assuming Flux is the correct class in model.py, renaming for clarity based on paper text

class ExpertModel(nn.Module):
    """
    Represents a single expert diffusion model in the DDM ensemble.
    This model is typically trained on a specific partition of the dataset.
    It wraps a pre-existing diffusion model architecture, like MMDiT (Flux).
    """
    def __init__(self, mmdit_config: dict):
        """
        Initializes the ExpertModel.

        Args:
            mmdit_config (dict): Configuration dictionary for the underlying MMDiT (Flux) model.
                                 This should contain parameters expected by the Flux class __init__
                                 (e.g., FluxParams or individual args like hidden_size, depth, etc.).
                                 It can also include a 'checkpoint_path' key for loading weights.
        """
        super().__init__()

        # Separate checkpoint path from model config keys
        checkpoint_path = mmdit_config.pop("checkpoint_path", None)

        # Instantiate the underlying diffusion model (MMDiT/Flux)
        # Ensure mmdit_config keys match the expected arguments of MMDiT.__init__
        # If MMDiT expects a FluxParams object, you might need to create it here:
        # from .flux.model import FluxParams
        # flux_params = FluxParams(**mmdit_config) # Assuming config keys match FluxParams fields
        # self.model = MMDiT(flux_params)
        # Or if it takes kwargs directly:
        try:
             self.model = MMDiT(**mmdit_config) # Assumes MMDiT.__init__ accepts these kwargs
        except TypeError as e:
             print(f"Error initializing MMDiT. Check if mmdit_config keys match MMDiT/Flux constructor arguments: {e}")
             print(f"Provided config keys: {list(mmdit_config.keys())}")
             # Potentially re-raise or handle differently
             raise e

        if checkpoint_path:
            self.load_weights(checkpoint_path)
        else:
            print("ExpertModel initialized without loading specific checkpoint.")


    def load_weights(self, checkpoint_path: str):
        """Loads weights from a checkpoint file."""
        try:
            # Load checkpoint onto CPU first to avoid GPU memory spikes
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Determine the actual state dict (common patterns: 'state_dict', 'model', raw dict)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                # If after checking common keys, it's still a dict but not the state_dict itself,
                # maybe the checkpoint *is* the state_dict
                if not isinstance(state_dict, dict): 
                     state_dict = checkpoint 
            elif isinstance(checkpoint, nn.Module): # If the entire model was saved
                 state_dict = checkpoint.state_dict()
            else:
                 raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}. Expected dict or nn.Module.")

            # Handle potential prefix issues (e.g., 'module.' from DataParallel/DDP)
            adjusted_state_dict = {}
            for k, v in state_dict.items():
                 name = k[len("module."):] if k.startswith("module.") else k # remove `module.` prefix
                 adjusted_state_dict[name] = v

            # Load the state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(adjusted_state_dict, strict=False)

            # Provide feedback on loading
            if not missing_keys and not unexpected_keys:
                 print(f"Expert weights loaded successfully and strictly from {checkpoint_path}")
            else:
                 if missing_keys:
                      print(f"Warning: Missing keys when loading expert weights from {checkpoint_path}: {missing_keys}")
                 if unexpected_keys:
                      print(f"Warning: Unexpected keys when loading expert weights from {checkpoint_path}: {unexpected_keys}")
                 print(f"Expert weights loaded non-strictly from {checkpoint_path}")

        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            # Depending on use case, might want to raise an error or allow initialization without weights
            # raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            print(f"Error loading expert weights from {checkpoint_path}: {e}")
            # Re-raise the exception for clearer debugging
            raise e


    def forward(self, *args, **kwargs):
        """
        Forward pass delegated to the underlying expert model (MMDiT/Flux).

        Accepts arguments expected by the underlying model's forward method.
        Common args for diffusion models:
            x (torch.Tensor): Input noisy tensor (e.g., latent image representation).
            t (torch.Tensor): Timestep tensor.
            y (torch.Tensor, optional): Conditioning tensor (e.g., text embeddings).
        Consult the specific MMDiT/Flux forward signature for exact requirements.

        Returns:
            torch.Tensor: The expert's prediction (e.g., predicted noise or x0).
        """
        # Delegate the forward pass directly to the wrapped model
        return self.model(*args, **kwargs)
