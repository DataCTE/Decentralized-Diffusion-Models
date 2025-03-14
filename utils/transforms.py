"""Image transformation utilities for Decentralized Diffusion Models."""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_train_transforms(config):
    """
    Get standard training transforms from config
    
    Args:
        config: Configuration object with transform settings
        
    Returns:
        Composed torchvision transforms
    """
    image_size = getattr(config, 'image_size', 512)
    mean = getattr(config, 'normalize_mean', [0.5, 0.5, 0.5])
    std = getattr(config, 'normalize_std', [0.5, 0.5, 0.5])
    
    return T.Compose([
        T.RandomResizedCrop(
            image_size, 
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def get_val_transforms(config):
    """
    Get standard validation transforms from config
    
    Args:
        config: Configuration object with transform settings
        
    Returns:
        Composed torchvision transforms
    """
    image_size = getattr(config, 'image_size', 512)
    mean = getattr(config, 'normalize_mean', [0.5, 0.5, 0.5])
    std = getattr(config, 'normalize_std', [0.5, 0.5, 0.5])
    
    return T.Compose([
        T.Resize(
            image_size,
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def get_inference_transforms(config):
    """
    Get standard inference transforms from config
    
    Args:
        config: Configuration object with transform settings
        
    Returns:
        Composed torchvision transforms
    """
    image_size = getattr(config, 'image_size', 512)
    mean = getattr(config, 'normalize_mean', [0.5, 0.5, 0.5])
    std = getattr(config, 'normalize_std', [0.5, 0.5, 0.5])
    
    return T.Compose([
        T.Resize(
            image_size,
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def denormalize(images):
    """
    Denormalize images from [-1, 1] to [0, 1]
    
    Args:
        images: Tensor of images in range [-1, 1]
        
    Returns:
        Tensor of images in range [0, 1]
    """
    return (images + 1) / 2

def normalize(images):
    """
    Normalize images from [0, 1] to [-1, 1]
    
    Args:
        images: Tensor of images in range [0, 1]
        
    Returns:
        Tensor of images in range [-1, 1]
    """
    return images * 2 - 1

def resize_image(image, size, keep_aspect_ratio=True):
    """
    Resize an image while optionally preserving aspect ratio
    
    Args:
        image: PIL Image or tensor
        size: Target size as (height, width) or single value for both
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    # Convert size to tuple if needed
    if isinstance(size, int):
        size = (size, size)
    
    # Handle PIL Image
    if isinstance(image, Image.Image):
        if keep_aspect_ratio:
            image.thumbnail(size[::-1], Image.BICUBIC)
            # Create new image with correct size and paste original
            new_image = Image.new("RGB", size[::-1], (0, 0, 0))
            new_image.paste(
                image, 
                ((size[1] - image.size[0]) // 2, (size[0] - image.size[1]) // 2)
            )
            return new_image
        else:
            return image.resize(size[::-1], Image.BICUBIC)
    
    # Handle tensor (assume BCHW or CHW)
    elif torch.is_tensor(image):
        if image.dim() == 3:  # CHW
            c, h, w = image.shape
            if keep_aspect_ratio:
                # Calculate scaling factor
                scale = min(size[0] / h, size[1] / w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # Resize with aspect ratio preserved
                resized = T.functional.resize(
                    image, 
                    [new_h, new_w], 
                    interpolation=T.InterpolationMode.BICUBIC
                )
                
                # Create new image and place resized one in center
                result = torch.zeros(c, size[0], size[1], device=image.device)
                start_h = (size[0] - new_h) // 2
                start_w = (size[1] - new_w) // 2
                result[:, start_h:start_h+new_h, start_w:start_w+new_w] = resized
                return result
            else:
                return T.functional.resize(
                    image, 
                    size, 
                    interpolation=T.InterpolationMode.BICUBIC
                )
        elif image.dim() == 4:  # BCHW
            # Apply to each image in batch
            return torch.stack([
                resize_image(img, size, keep_aspect_ratio) 
                for img in image
            ])
    
    # Handle numpy array
    elif isinstance(image, np.ndarray):
        # Convert to tensor, resize, then back to numpy
        if image.shape[0] == 3:  # CHW
            tensor = torch.from_numpy(image)
            return resize_image(tensor, size, keep_aspect_ratio).numpy()
        else:  # HWC
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            result = resize_image(tensor, size, keep_aspect_ratio).permute(1, 2, 0).numpy()
            return result
    
    raise ValueError(f"Unsupported image type: {type(image)}")

def pad_to_multiple(image, multiple=8):
    """
    Pad image to make dimensions multiple of a specific value
    
    Args:
        image: Image tensor in BCHW or CHW format
        multiple: Dimensions will be multiples of this value
        
    Returns:
        Padded image
    """
    # Get dimensions
    if image.dim() == 4:  # BCHW
        _, _, h, w = image.shape
    else:  # CHW
        _, h, w = image.shape
    
    # Calculate padding
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # Pad image
    padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
    return T.functional.pad(image, padding, padding_mode='constant')

def center_crop_tensor(tensor, size):
    """
    Center crop a tensor
    
    Args:
        tensor: Image tensor in BCHW or CHW format
        size: Crop size as (height, width) or single value for both
        
    Returns:
        Cropped tensor
    """
    # Convert size to tuple if needed
    if isinstance(size, int):
        size = (size, size)
    
    # Get current dimensions
    if tensor.dim() == 4:  # BCHW
        _, _, h, w = tensor.shape
    else:  # CHW
        _, h, w = tensor.shape
    
    # Calculate crop coordinates
    top = (h - size[0]) // 2
    left = (w - size[1]) // 2
    
    # Crop tensor
    return T.functional.crop(tensor, top, left, size[0], size[1])

def random_crop_tensor(tensor, size):
    """
    Random crop a tensor
    
    Args:
        tensor: Image tensor in BCHW or CHW format
        size: Crop size as (height, width) or single value for both
        
    Returns:
        Cropped tensor
    """
    # Convert size to tuple if needed
    if isinstance(size, int):
        size = (size, size)
    
    # Get current dimensions
    if tensor.dim() == 4:  # BCHW
        _, _, h, w = tensor.shape
    else:  # CHW
        _, h, w = tensor.shape
    
    # Calculate max crop coordinates
    max_top = h - size[0]
    max_left = w - size[1]
    
    # Get random crop coordinates
    top = torch.randint(0, max(1, max_top + 1), (1,)).item()
    left = torch.randint(0, max(1, max_left + 1), (1,)).item()
    
    # Crop tensor
    return T.functional.crop(tensor, top, left, size[0], size[1])

# Create transform presets
def create_transform_presets(config):
    """
    Create a dictionary of transform presets
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of transform presets
    """
    return {
        'train': get_train_transforms(config),
        'val': get_val_transforms(config),
        'inference': get_inference_transforms(config)
    } 