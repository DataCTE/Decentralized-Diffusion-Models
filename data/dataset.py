"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import logging
import time
import glob
import math
import random
import io
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union, Set

# Import centralized utilities
from utils.distributed import is_main_process, get_rank, get_world_size, broadcast_object, synchronize
from utils.logging import setup_logger
from utils.transforms import get_train_transforms, get_val_transforms, resize_image, normalize
from utils.visualization import tensor_to_pil

# Setup logging
logger = logging.getLogger(__name__)

class DataValidator:
    """Shared image validation utility to avoid duplicate validation between dataset classes"""
    
    _valid_files_cache = {}  # Cache for valid files
    _invalid_files_cache = set()  # Cache for known invalid files
    _validated = False  # Flag to track if validation has been performed
    
    @classmethod
    def validate_images(cls, root_dir, extensions=None, min_size=256):
        """
        Validate all images in a directory, with results shared between processes
        
        Args:
            root_dir: Directory containing images
            extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png', '.webp'])
            min_size: Minimum image size to be considered valid
            
        Returns:
            List of valid image files
        """
        # Check if we've already validated this directory
        cache_key = f"{root_dir}_{str(extensions)}_{min_size}"
        if cache_key in cls._valid_files_cache:
            return cls._valid_files_cache[cache_key]
            
        # Set default extensions if none provided
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
            
        # Only validate on main process (rank 0) to avoid duplicate work
        if is_main_process():
            logger.info(f"Validating images in {root_dir}")
            start_time = time.time()
            valid_files = []
            invalid_count = 0
            
            # Get all image files in the directory
            all_files = []
            for ext in extensions:
                all_files.extend(glob.glob(os.path.join(root_dir, f"*{ext}")))
                all_files.extend(glob.glob(os.path.join(root_dir, f"*{ext.upper()}")))
            
            # Check validity of each file
            for img_path in all_files:
                # Skip if already known to be invalid
                if img_path in cls._invalid_files_cache:
                    invalid_count += 1
                    continue
                    
                # Check if file is valid
                if cls.is_valid_image(img_path, min_size):
                    valid_files.append(img_path)
                else:
                    cls._invalid_files_cache.add(img_path)
                    invalid_count += 1
            
            # Store valid files in cache
            elapsed = time.time() - start_time
            logger.info(f"Image validation completed in {elapsed:.2f}s: {len(valid_files)} valid, {invalid_count} invalid")
            cls._valid_files_cache[cache_key] = valid_files
            result = valid_files
        else:
            # Non-main processes get an empty list initially
            result = []
        
        # Broadcast result from main process to all others
        result = broadcast_object(result, src=0)
        
        # Update cache on all processes
        if cache_key not in cls._valid_files_cache:
            cls._valid_files_cache[cache_key] = result
            
        cls._validated = True
        return result
    
    @staticmethod
    def is_valid_image(img_path, min_size=256):
        """
        Check if an image is valid
        
        Args:
            img_path: Path to image file
            min_size: Minimum dimension size
            
        Returns:
            Boolean indicating if image is valid
        """
        if not os.path.exists(img_path):
            return False
            
        try:
            # Attempt to open the image
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                
            # Verify image can be opened and has valid dimensions
            with Image.open(io.BytesIO(img_bytes)) as img:
                width, height = img.size
                
                # Check minimum dimensions
                if width < min_size or height < min_size:
                    return False
                    
                # Check if image has color channels (not grayscale)
                if img.mode not in ('RGB', 'RGBA'):
                    return False
            
            return True
        except Exception:
            # Any error means the image is invalid
            return False

class DDMDataset(Dataset):
    """Implementation of dataset from paper with batching by aspect ratio"""
    
    def __init__(self, root_dir, transform=None, cluster_labels=None, include_metadata=True):
        """
        Initialize dataset with cluster assignments from ClusterManager
        
        Args:
            root_dir: Directory containing images
            transform: Optional transforms to apply
            cluster_labels: Pre-computed cluster labels from ClusterManager
            include_metadata: Whether to include image metadata
        """
        self.root_dir = root_dir
        
        # Initialize logger
        self.logger = setup_logger(name="DDMDataset", rank=get_rank())
        
        # Set transform or use default
        self.transform = transform
        
        # Store other parameters
        self.include_metadata = include_metadata
        
        # Initialize file paths
        self.image_files = DataValidator.validate_images(root_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
            
        self.logger.info(f"Found {len(self.image_files)} valid images")
        
        # Collect image sizes for bucket batching
        self.image_sizes = {}
        self._collect_image_sizes()
        
        # Generate dynamic buckets based on aspect ratios
        self.buckets = []
        self._generate_dynamic_buckets()
        
        # Assign images to buckets
        self.bucket_indices = defaultdict(list)
        self.assign_buckets()
        
        # Initialize cluster assignments
        if cluster_labels is not None:
            self._init_shared_clusters(cluster_labels)
        else:
            self.cluster_assignments = [-1] * len(self.image_files)  # Default: unassigned
            
        # Initialize bucket assignments
        self._init_bucket_assignments()
        
        self.logger.info(f"Initialized dataset with {len(self.image_files)} images across {len(self.buckets)} buckets")
        
    def _init_shared_clusters(self, cluster_labels):
        """
        Initialize cluster assignments from ClusterManager
        
        Args:
            cluster_labels: Pre-computed cluster labels
        """
        # Ensure correct length
        if len(cluster_labels) != len(self.image_files):
            self.logger.warning(f"Cluster labels length ({len(cluster_labels)}) does not match dataset length ({len(self.image_files)})")
            
            # Handle mismatch
            if len(cluster_labels) > len(self.image_files):
                cluster_labels = cluster_labels[:len(self.image_files)]
            else:
                # Pad with -1 (unassigned)
                cluster_labels = np.concatenate([
                    cluster_labels, 
                    np.full(len(self.image_files) - len(cluster_labels), -1)
                ])
        
        self.cluster_assignments = cluster_labels
        
    def _init_bucket_assignments(self):
        """Initialize bucket assignments for efficient batching"""
        # Map bucket indices to original indices
        self.bucket_assignments = {}
        for bucket_idx, indices in self.bucket_indices.items():
            for i, idx in enumerate(indices):
                self.bucket_assignments[idx] = (bucket_idx, i)

    def _collect_image_sizes(self):
        """Collect image sizes for bucket batching"""
        if is_main_process():
            self.logger.info("Collecting image sizes for bucket batching")
            
        # Process images in chunks for memory efficiency
        chunk_size = 1000
        num_chunks = (len(self.image_files) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(self.image_files))
            
            for idx in tqdm(range(start_idx, end_idx), 
                           desc=f"Collecting sizes {i+1}/{num_chunks}",
                           disable=not is_main_process()):
                path = self.image_files[idx]
                
                try:
                    with Image.open(path) as img:
                        self.image_sizes[path] = (img.width, img.height)
                except Exception as e:
                    self.logger.warning(f"Failed to get size for {path}: {str(e)}")
                    
    def _generate_dynamic_buckets(self, num_buckets=20):
        """
        Generate dynamic buckets based on aspect ratios
        
        Args:
            num_buckets: Number of buckets to generate
        """
        if not self.image_sizes:
            self.logger.warning("No image sizes collected, can't generate buckets")
            return
            
        # Collect aspect ratios
        aspect_ratios = []
        for _, (width, height) in self.image_sizes.items():
            aspect_ratios.append(width / height)
            
        if not aspect_ratios:
            return
            
        # Generate buckets
        min_ar = min(aspect_ratios)
        max_ar = max(aspect_ratios)
        
        # Use logarithmic spacing for better distribution
        log_min = np.log(min_ar)
        log_max = np.log(max_ar)
        
        # Generate bucket boundaries
        bucket_boundaries = np.exp(np.linspace(log_min, log_max, num_buckets + 1))
        
        # Create buckets as (min_ar, max_ar, target_ar)
        self.buckets = []
        for i in range(len(bucket_boundaries) - 1):
            min_bucket_ar = bucket_boundaries[i]
            max_bucket_ar = bucket_boundaries[i + 1]
            target_ar = (min_bucket_ar + max_bucket_ar) / 2
            self.buckets.append((min_bucket_ar, max_bucket_ar, target_ar))
            
    def find_closest_bucket(self, width, height):
        """
        Find the closest bucket for a given image size
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Index of the closest bucket
        """
        if not self.buckets:
            return 0  # No buckets, use default
            
        aspect_ratio = width / height
        
        # Find bucket with exact range match
        for i, (min_ar, max_ar, _) in enumerate(self.buckets):
            if min_ar <= aspect_ratio <= max_ar:
                return i
                
        # If no exact match, find closest bucket
        closest_bucket = 0
        min_distance = float('inf')
        
        for i, (_, _, target_ar) in enumerate(self.buckets):
            distance = abs(target_ar - aspect_ratio)
            if distance < min_distance:
                min_distance = distance
                closest_bucket = i
                
        return closest_bucket
        
    def assign_buckets(self):
        """Assign images to buckets based on aspect ratio"""
        if is_main_process():
            self.logger.info("Assigning images to aspect ratio buckets")
            
        # Reset bucket indices
        self.bucket_indices = defaultdict(list)
        
        # Assign images to buckets
        for idx, path in enumerate(self.image_files):
            if path in self.image_sizes:
                width, height = self.image_sizes[path]
                bucket_idx = self.find_closest_bucket(width, height)
                self.bucket_indices[bucket_idx].append(idx)
            else:
                # Default bucket for unknown sizes
                self.bucket_indices[0].append(idx)
                
        # Log bucket distribution
        if is_main_process():
            bucket_sizes = {i: len(indices) for i, indices in self.bucket_indices.items()}
            total_images = sum(bucket_sizes.values())
            
            for bucket_idx, size in bucket_sizes.items():
                if bucket_idx < len(self.buckets):
                    min_ar, max_ar, target_ar = self.buckets[bucket_idx]
                    self.logger.info(f"Bucket {bucket_idx}: {size} images ({size/total_images*100:.1f}%) - AR [{min_ar:.2f}-{max_ar:.2f}], Target: {target_ar:.2f}")
                else:
                    self.logger.info(f"Bucket {bucket_idx}: {size} images ({size/total_images*100:.1f}%) - Unknown AR range")
                
    def __len__(self):
        """Return the number of valid images"""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """Get a dataset item with its cluster assignment"""
        # Get image path
        path = self.image_files[idx]
        
        # Load image
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            self.logger.warning(f"Failed to load image {path}: {str(e)}")
            # Return a small black image as fallback
            image = Image.new('RGB', (64, 64), color=0)
            
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Use centralized transforms
            transform = get_train_transforms(self.config if hasattr(self, 'config') else None)
            image = transform(image)
            
        # Get cluster assignment
        cluster = self.cluster_assignments[idx]
        
        # Create return dictionary
        result = {
            'image': image,
            'cluster': cluster,
            'idx': idx
        }
        
        # Add metadata if requested
        if self.include_metadata:
            result['path'] = path
            
            # Get bucket assignment if available
            if hasattr(self, 'bucket_assignments') and idx in self.bucket_assignments:
                result['bucket'] = self.bucket_assignments[idx]
                
        return result

class FeatureDataset(Dataset):
    """Dataset for extracting features for clustering"""
    
    def __init__(self, root_dir, config=None):
        """
        Initialize dataset for feature extraction
        
        Args:
            root_dir: Directory containing images
            config: Configuration object
        """
        self.root_dir = root_dir
        self.config = config
        
        # Initialize logger
        self.logger = setup_logger(name="FeatureDataset", rank=get_rank())
        
        # Get image size from config or use default
        if config is not None:
            self.image_size = getattr(config, 'feature_image_size', 224)
        else:
            self.image_size = 224
            
        # Initialize file paths
        self.image_files = DataValidator.validate_images(root_dir)
        self.valid_indices = []
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid images found in {root_dir}")
            
        self.logger.info(f"Found {len(self.image_files)} valid images in {root_dir}")
            
        # Set transform for feature extraction
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        """Return the number of valid images"""
        return len(self.image_files)
        
    def __getitem__(self, idx):
        """Get a dataset item for feature extraction"""
        # Get image path
        path = self.image_files[idx]
        
        # Load image
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            self.logger.warning(f"Failed to load image {path}: {str(e)}")
            # Return a small black image as fallback
            image = Image.new('RGB', (64, 64), color=0)
            
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Resize using centralized transform
            image = resize_image(image, (self.image_size, self.image_size))
            
            # Convert to tensor and normalize
            if isinstance(image, Image.Image):
                image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
                image = normalize(image)
            
        # Create return dictionary
        result = {
            'image': image,
            'path': path,
            'idx': idx
        }
        
        return result

class BucketBatchSampler:
    """Sampler that creates batches from image buckets of similar aspect ratios"""
    
    def __init__(self, bucket_indices, batch_size, shuffle=True):
        """
        Initialize bucket sampler
        
        Args:
            bucket_indices: List of indices for each bucket
            batch_size: Batch size
            shuffle: Whether to shuffle indices
        """
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate number of batches
        self.num_batches = sum([len(indices) // batch_size + (1 if len(indices) % batch_size > 0 else 0)
                              for indices in bucket_indices.values()])
        
    def __iter__(self):
        # Shuffle all samples if required
        if self.shuffle:
            for bucket in self.bucket_indices:
                np.random.shuffle(self.bucket_indices[bucket])
        
        # Create batches from buckets
        batches = []
        
        # Process each bucket
        for bucket_idx in self.bucket_indices:
            indices = self.bucket_indices[bucket_idx]
            
            # Skip empty buckets
            if not indices:
                continue
                
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                # If batch is smaller than batch_size and not the last one, borrow from next bucket
                if len(batch) < self.batch_size and i + self.batch_size < len(indices):
                    extra = indices[i + self.batch_size:i + 2 * self.batch_size]
                    batch.extend(extra[:self.batch_size - len(batch)])
                    
                batches.append(batch)
                
        # Shuffle batches if required
        if self.shuffle:
            np.random.shuffle(batches)
            
        # Yield batches
        for batch in batches:
            yield batch
            
    def __len__(self):
        """Return the number of batches"""
        return self.num_batches

def create_expert_bucket_loaders(dataset, config, world_size=1, rank=0):
    """
    Create separate loaders for each expert's assigned data
    
    Args:
        dataset: Dataset with cluster assignments
        config: Configuration object
        world_size: Number of processes
        rank: Current process rank
        
    Returns:
        Dictionary mapping expert index to DataLoader
    """
    logger = setup_logger(name="ExpertLoaders", rank=rank)
    logger.info("Creating per-expert data loaders with bucket batching")
    
    # Initialize per-expert loaders
    expert_loaders = {}
    
    # Get all indices assigned to each expert
    expert_indices = {}
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        cluster_idx = item['cluster']
        
        # Skip unassigned samples
        if cluster_idx < 0:
            continue
            
        if cluster_idx not in expert_indices:
            expert_indices[cluster_idx] = []
            
        expert_indices[cluster_idx].append(idx)
        
    # Create bucket indices for each expert
    expert_bucket_indices = {}
    
    for expert_idx, indices in expert_indices.items():
        bucket_indices = defaultdict(list)
        
        for idx in indices:
            item = dataset[idx]
            
            # Get bucket if available, otherwise use default
            if 'bucket' in item:
                bucket_idx = item['bucket'][0]  # (bucket_idx, position)
            else:
                bucket_idx = 0
                
            bucket_indices[bucket_idx].append(idx)
            
        expert_bucket_indices[expert_idx] = bucket_indices
        
    # Create loaders for each expert
    for expert_idx, bucket_indices in expert_bucket_indices.items():
        # Create sampler
        sampler = BucketBatchSampler(
            bucket_indices=bucket_indices,
            batch_size=config.expert_batch_size,
            shuffle=True
        )
        
        # Create loader
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=getattr(config, 'pin_memory', True),
            persistent_workers=getattr(config, 'persistent_workers', True)
        )
        
        expert_loaders[expert_idx] = loader
        
        logger.info(f"Created loader for expert {expert_idx} with {len(sampler)} batches")
        
    return expert_loaders 