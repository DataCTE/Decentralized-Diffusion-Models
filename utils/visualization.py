"""Visualization utilities for Decentralized Diffusion Models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import io
import logging
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)

def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL Image
    
    Args:
        tensor: Image tensor in range [-1, 1] or [0, 1]
        
    Returns:
        PIL Image
    """
    # Handle batched tensors
    if tensor.dim() == 4:
        return [tensor_to_pil(t) for t in tensor]
    
    # Ensure tensor is on CPU
    tensor = tensor.cpu().detach()
    
    # Scale tensor to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Convert to numpy and scale to [0, 255]
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    return Image.fromarray(array)

def pil_to_tensor(image, normalize=True):
    """
    Convert a PIL Image to a tensor
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Image tensor
    """
    # Convert to numpy
    array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    
    # Normalize to [-1, 1] if requested
    if normalize:
        tensor = tensor * 2 - 1
    
    return tensor

def create_image_grid(images, nrow=None):
    """
    Create a grid of images
    
    Args:
        images: List of PIL Images or tensor of shape [B, C, H, W]
        nrow: Number of images per row
        
    Returns:
        PIL Image containing the grid
    """
    # Convert to tensors if needed
    if isinstance(images, list) and isinstance(images[0], Image.Image):
        tensors = [pil_to_tensor(img, normalize=False) for img in images]
        images_tensor = torch.stack(tensors)
    elif torch.is_tensor(images):
        # Ensure tensor is in range [0, 1]
        if images.min() < 0:
            images_tensor = (images + 1) / 2
        else:
            images_tensor = images
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")
    
    # Determine grid size
    if nrow is None:
        nrow = int(math.sqrt(images_tensor.size(0)))
    
    # Create grid
    grid = make_grid(images_tensor, nrow=nrow, padding=2, normalize=False)
    
    # Convert to PIL Image
    return tensor_to_pil(grid)

def plot_images(images, titles=None, figsize=(12, 12), nrows=None, ncols=None):
    """
    Plot a set of images in a grid
    
    Args:
        images: List of PIL Images, tensors, or numpy arrays
        titles: Optional list of titles for each image
        figsize: Figure size
        nrows: Number of rows (calculated from len(images) if None)
        ncols: Number of columns (calculated from len(images) if None)
        
    Returns:
        Figure and axes
    """
    # Calculate grid dimensions
    n_images = len(images)
    if nrows is None and ncols is None:
        ncols = int(math.sqrt(n_images))
        nrows = math.ceil(n_images / ncols)
    elif nrows is None:
        nrows = math.ceil(n_images / ncols)
    elif ncols is None:
        ncols = math.ceil(n_images / nrows)
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Convert all images to numpy arrays
    for i, img in enumerate(images):
        if i >= len(axes):
            break
            
        # Convert to numpy array based on type
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        elif torch.is_tensor(img):
            # Handle different tensor formats
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
            
            if img.dim() == 3:
                # Ensure tensor is in range [0, 1]
                if img.min() < 0:
                    img = (img + 1) / 2
                img_array = img.cpu().detach().permute(1, 2, 0).numpy()
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img.cpu().detach().numpy()
        elif isinstance(img, np.ndarray):
            img_array = img
        else:
            logger.warning(f"Unsupported image type: {type(img)}, skipping")
            continue
        
        # Plot image
        axes[i].imshow(img_array)
        axes[i].axis('off')
        
        # Add title if provided
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig, axes

def fig_to_image(fig):
    """
    Convert a matplotlib figure to a PIL Image
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        PIL Image
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    
    # Convert buffer to PIL Image
    img = Image.open(buf)
    return img

def visualize_embeddings(embeddings, labels=None, method='pca', figsize=(10, 10)):
    """
    Visualize embeddings in 2D
    
    Args:
        embeddings: [N, D] tensor or array of embeddings
        labels: Optional [N] tensor or array of labels
        method: Dimensionality reduction method ('pca' or 'tsne')
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    # Ensure embeddings are numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().detach().numpy()
    
    # Ensure labels are numpy if provided
    if labels is not None and torch.is_tensor(labels):
        labels = labels.cpu().detach().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot embeddings
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            c=labels, 
            cmap='tab10',
            alpha=0.7
        )
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
    else:
        ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1], 
            alpha=0.7
        )
    
    ax.set_title(f"Embeddings visualized with {method.upper()}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(alpha=0.3)
    
    return fig, ax

def plot_attention_map(attention, tokens=None, figsize=(10, 10)):
    """
    Visualize attention map
    
    Args:
        attention: [N, N] tensor or array of attention weights
        tokens: Optional list of tokens for axis labels
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    # Ensure attention is numpy
    if torch.is_tensor(attention):
        attention = attention.cpu().detach().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot attention
    im = ax.imshow(attention, cmap='viridis')
    
    # Add colorbar
    fig.colorbar(im, ax=ax)
    
    # Add token labels if provided
    if tokens is not None:
        # Ensure tokens is a list of strings
        tokens = [str(token) for token in tokens]
        
        # Add labels
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
    
    ax.set_title("Attention Map")
    
    return fig, ax

def plot_loss_curves(losses, figsize=(10, 6)):
    """
    Plot training and validation loss curves
    
    Args:
        losses: Dictionary of losses (e.g., {'train': [...], 'val': [...]})
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot losses
    for name, values in losses.items():
        ax.plot(values, label=name)
    
    ax.set_title("Loss Curves")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig, ax

def plot_image_sequences(sequences, titles=None, figsize=None):
    """
    Plot sequences of images (e.g., diffusion steps)
    
    Args:
        sequences: List of sequences, where each sequence is a list of images
        titles: Optional list of titles for each sequence
        figsize: Figure size
        
    Returns:
        Figure and axes
    """
    n_sequences = len(sequences)
    seq_length = len(sequences[0])
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (3 * seq_length, 3 * n_sequences)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_sequences, seq_length, figsize=figsize)
    if n_sequences == 1:
        axes = axes.reshape(1, -1)
    
    # Plot sequences
    for i, sequence in enumerate(sequences):
        for j, img in enumerate(sequence):
            # Convert to numpy array based on type
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            elif torch.is_tensor(img):
                # Handle different tensor formats
                if img.dim() == 4 and img.size(0) == 1:
                    img = img.squeeze(0)
                
                if img.dim() == 3:
                    # Ensure tensor is in range [0, 1]
                    if img.min() < 0:
                        img = (img + 1) / 2
                    img_array = img.cpu().detach().permute(1, 2, 0).numpy()
                else:
                    img_array = img.cpu().detach().numpy()
            elif isinstance(img, np.ndarray):
                img_array = img
            else:
                logger.warning(f"Unsupported image type: {type(img)}, skipping")
                continue
            
            # Plot image
            axes[i, j].imshow(img_array)
            axes[i, j].axis('off')
            
            # Add sequence title
            if j == 0 and titles is not None and i < len(titles):
                axes[i, j].set_title(titles[i], loc='left')
    
    plt.tight_layout()
    return fig, axes 