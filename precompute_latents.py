#!/usr/bin/env python3
"""Precompute VAE latents and CLIP embeddings for Decentralized Diffusion Models."""

import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import get_config
from data.vae import VAEWrapper
from data.clip import CLIPTextEncoder  # Import CLIPTextEncoder
from data.dataset import DDMDataset
from utils.distributed import setup_distributed, get_rank, get_world_size, is_main_process
from datetime import timedelta
import io
from PIL import Image
import argparse  # Import argparse for command-line arguments
import torchvision.transforms as transforms # Import torchvision transforms
from data.transforms import normalize # Import normalize function

def precompute_latents(config_path="config.py", output_dir="cache", precompute_vae=True, precompute_clip=True):
    """Main function to precompute VAE latents and CLIP embeddings using distributed processing."""

    # Distributed setup
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank, world_size, device = distributed_setup()
    else:
        rank, world_size, device = single_process_setup()

    config = get_config(config_path)
    dataset_config = config  # Use the same config for dataset

    vae = None
    clip = None

    # Ensure output directories exist
    if precompute_vae:
        vae_latent_output_dir = os.path.join(output_dir, "latents")
        os.makedirs(vae_latent_output_dir, exist_ok=True)
        # Load VAE model only if precomputing VAE latents
        vae = VAEWrapper(device, config)
        vae.vae.eval()  # Set VAE to evaluation mode
    else:
        vae_latent_output_dir = None

    if precompute_clip:
        clip_embedding_output_dir = os.path.join(output_dir, "clip_embeddings")
        os.makedirs(clip_embedding_output_dir, exist_ok=True)
        # Load CLIP model only if precomputing CLIP embeddings
        clip = CLIPTextEncoder(device, config)
        clip.text_encoder.eval() # Set CLIP to eval mode
    else:
        clip_embedding_output_dir = None


    # Initialize dataset (only for file discovery)
    dataset = DDMDataset(dataset_config, split='train', bypass_clip_check=True, bypass_latent_check=True)

    image_files_partition = partition_data(dataset.image_files, rank, world_size)

    # Process images and save latents and CLIP embeddings
    process_and_save_latents(
        image_files_partition, device, output_dir, vae, clip, rank, world_size, config,
        vae_latent_output_dir=vae_latent_output_dir, clip_embedding_output_dir=clip_embedding_output_dir,
        precompute_vae=precompute_vae, precompute_clip=precompute_clip
    )

def distributed_setup():
    """Configure distributed training environment"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=timedelta(minutes=90)
    )

    world_size = dist.get_world_size()
    print(f"Rank {rank}: Using device: {device}")
    return rank, world_size, device

def single_process_setup():
    """Configure single process environment"""
    rank, world_size = 0, 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"Using device: {device}")
    return rank, world_size, device

def partition_data(data, rank, world_size):
    """Split data evenly across processes"""
    if not data:
        return []
    partition_size = len(data) // world_size
    start = rank * partition_size
    end = start + partition_size
    if rank == world_size - 1:
        end = len(data)
    return data[start:end]

def process_and_save_latents(image_files, device, output_dir, vae, clip, rank, world_size, config, vae_latent_output_dir=None, clip_embedding_output_dir=None, precompute_vae=True, precompute_clip=True):
    """Process images, encode to latents, and save."""
    latent_extension = ".latent.pt"
    clip_embedding_extension = ".clip_emb.pt" # New: extension for CLIP embeddings
    processor_count = min(8, os.cpu_count()) # Adjust as needed

    with ThreadPoolExecutor(max_workers=processor_count) as executor:
        futures = [executor.submit(
                       process_single_image, img_path, device, output_dir, vae, clip,
                       latent_extension, clip_embedding_extension, config,
                       vae_latent_output_dir, clip_embedding_output_dir,
                       precompute_vae, precompute_clip
                   ) for img_path in image_files]

        desc = f"Rank {rank} Encoding"
        if precompute_vae and precompute_clip:
            desc += " Latents and CLIP Embeddings"
        elif precompute_vae:
            desc += " Latents"
        elif precompute_clip:
            desc += " CLIP Embeddings"
        else:
            desc += " (Nothing to precompute)"

        with tqdm(total=len(image_files), desc=desc, position=rank) as pbar:
            for future in as_completed(futures):
                future.result() # Get result (or exception if raised)
                pbar.update(1)

def process_single_image(image_path, device, output_dir, vae, clip, latent_extension, clip_embedding_extension, config, vae_latent_output_dir=None, clip_embedding_output_dir=None, precompute_vae=True, precompute_clip=True):
    """Process one image: load, encode, save latent and/or CLIP embedding."""
    try:
        with open(image_path, 'rb') as f: # Open in binary mode
            img = Image.open(io.BytesIO(f.read())).convert('RGB') # Open from bytes
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            img_tensor = normalize(img_tensor) # Normalize to [-1, 1]

            if precompute_vae:
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=config.use_mixed_precision):
                    latents = vae.encode(img_tensor)

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                latent_file_path = os.path.join(vae_latent_output_dir, base_name + latent_extension)
                torch.save(latents.cpu(), latent_file_path) # Save latents to CPU in float32

            if precompute_clip:
                caption_path = os.path.splitext(image_path)[0] + ".txt" # Assuming .txt captions
                if os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as caption_file:
                        caption_text = caption_file.read().strip()
                else:
                    caption_text = "" # Default to empty string if no caption file

                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=config.use_mixed_precision):
                    clip_embeddings = clip.encode([caption_text]) # Encode caption text

                base_name = os.path.splitext(os.path.basename(image_path))[0]
                clip_embedding_file_path = os.path.join(clip_embedding_output_dir, base_name + clip_embedding_extension)
                torch.save(clip_embeddings.cpu(), clip_embedding_file_path) # Save CLIP embeddings to CPU

    except Exception as e:
        print(f"Rank: {get_rank()} Error processing image {image_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute VAE latents and CLIP embeddings.")
    parser.add_argument("--config_path", type=str, default="config.py", help="Path to config.py file.")
    parser.add_argument("--output_dir", type=str, default="cache", help="Output directory for cache.")
    parser.add_argument("--precompute_vae", action="store_true", help="Precompute VAE latents.")
    parser.add_argument("--precompute_clip", action="store_true", help="Precompute CLIP embeddings.")

    args = parser.parse_args()

    precompute_latents(
        config_path=args.config_path,
        output_dir=args.output_dir,
        precompute_vae=args.precompute_vae,
        precompute_clip=args.precompute_clip
    ) 