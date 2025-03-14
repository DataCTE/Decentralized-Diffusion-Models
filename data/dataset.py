"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import logging
import time
import glob

import random
import io
import torchvision.transforms as transforms

# Import centralized utilities
from utils.distributed import is_main_process, get_rank, broadcast_object
from utils.logging import setup_logger
from data.transforms import resize_image, normalize
from utils.distributed import synchronize

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

    @classmethod
    def validate_images_batch(cls, file_paths, min_size=256, max_workers=8):
        """
        Validate a batch of images efficiently using multiprocessing
        
        Args:
            file_paths: List of image file paths
            min_size: Minimum image size to be considered valid
            max_workers: Maximum number of worker processes
            
        Returns:
            (valid_files, invalid_files): Tuple of lists
        """
        import concurrent.futures
        from functools import partial
        
        # Function to validate a single image
        def _validate_single(img_path, min_size):
            if img_path in cls._invalid_files_cache:
                return (False, img_path)
                
            if img_path in cls._valid_files_cache:
                return (True, img_path)
                
            try:
                # Use PIL's lazy loading for faster validation
                with Image.open(img_path) as img:
                    # Validate image dimensions
                    if img.width < min_size or img.height < min_size:
                        cls._invalid_files_cache.add(img_path)
                        return (False, img_path)
                        
                    # Check if image can be fully loaded
                    img.load()
                    
                    # Valid image
                    return (True, img_path)
            except Exception as e:
                # Invalid image
                cls._invalid_files_cache.add(img_path)
                return (False, img_path)
        
        # Set up number of workers (limit to CPU count)
        import os
        cpu_count = os.cpu_count() or 4
        workers = min(max_workers, cpu_count)
        
        # Process images in parallel
        valid_files = []
        invalid_files = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            validator_fn = partial(_validate_single, min_size=min_size)
            for is_valid, img_path in executor.map(validator_fn, file_paths):
                if is_valid:
                    valid_files.append(img_path)
                    # Add to cache
                    cls._valid_files_cache[img_path] = True
                else:
                    invalid_files.append(img_path)
                    
        return valid_files, invalid_files
        
    @classmethod
    def validate_images_efficient(cls, root_dir, extensions=None, min_size=256, batch_size=1000):
        """
        Efficiently validate all images in a directory with batched processing and multiprocessing
        
        Args:
            root_dir: Directory containing images
            extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png', '.webp'])
            min_size: Minimum image size to be considered valid
            batch_size: Number of images to process in a batch
            
        Returns:
            List of valid image files
        """
        # Set default extensions if none provided
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
            
        # Only validate on main process (rank 0) to avoid duplicate work
        if not is_main_process():
            # Wait for rank 0 to finish validation
            synchronize()
            # Get result from rank 0
            return broadcast_object([], src=0)
            
        logger.info(f"Efficiently validating images in {root_dir}")
        start_time = time.time()
        
        # Get all image files in the directory
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(root_dir, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(root_dir, f"*{ext.upper()}")))
            
        # Process in batches
        all_valid_files = []
        total_processed = 0
        
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            valid_batch, invalid_batch = cls.validate_images_batch(batch, min_size=min_size)
            
            all_valid_files.extend(valid_batch)
            total_processed += len(batch)
            
            # Log progress
            if total_processed % 10000 == 0 or total_processed == len(all_files):
                elapsed = time.time() - start_time
                logger.info(f"Validated {total_processed}/{len(all_files)} images in {elapsed:.2f}s. "
                           f"{len(all_valid_files)} valid so far.")
                
        # Final timing
        elapsed = time.time() - start_time
        invalid_count = len(all_files) - len(all_valid_files)
        logger.info(f"Image validation completed in {elapsed:.2f}s: "
                   f"{len(all_valid_files)} valid, {invalid_count} invalid")
                
        # Synchronize after validation
        synchronize()
        
        # Broadcast to other processes
        return broadcast_object(all_valid_files, src=0)

class DDMDataset(Dataset):
    """Implementation of dataset from paper with batching by aspect ratio"""
    
    def __init__(self, config, split='train', transforms=None, cluster_labels=None, hf_split=None):
        """
        Initialize dataset with cluster information
        
        Args:
            config: Config object with dataset parameters
            split: Dataset split ('train', 'val', etc.)
            transforms: Image transformations to apply
            cluster_labels: Precomputed cluster labels for each image
            hf_split: HuggingFace dataset split name if different from 'split'
        """
        self.config = config
        self.split = split
        self.hf_split = hf_split or split
        self.transforms = transforms
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize path and data
        self.dataset_path = getattr(config, 'dataset_path', None)
        self.image_files = []
        self.captions = []
        self.cluster_assignments = None
        self.bucket_indices = {}  # Maps bucket index to list of indices
        self.bucket_assignments = {}  # Maps sample index to (bucket_idx, position)
        
        # Load dataset
        self._load_dataset()
        
        # Initialize clustering if provided
        if cluster_labels is not None:
            self._init_shared_clusters(cluster_labels)
        
        # Initialize buckets for efficient batching by aspect ratio
        self._init_buckets()
        
        # Initialize bucket assignments for faster lookup
        self._init_bucket_assignments()
        
        # Log dataset stats
        self.logger.info(f"Initialized {split} dataset with {len(self.image_files)} samples")
        if self.cluster_assignments is not None:
            valid_count = (self.cluster_assignments >= 0).sum()
            self.logger.info(f"Dataset has {valid_count}/{len(self.cluster_assignments)} valid cluster assignments")
    
    def _load_dataset(self):
        """Load dataset images and captions"""
        # Use HuggingFace datasets if configured
        if hasattr(self.config, 'use_hf_dataset') and self.config.use_hf_dataset:
            self._load_from_huggingface()
        else:
            self._load_from_files()
    
    def _load_from_files(self):
        """Load dataset from image files"""
        if self.dataset_path is None:
            raise ValueError("Dataset path not specified in config")
            
        # Build image list
        split_path = os.path.join(self.dataset_path, self.split)
        if not os.path.exists(split_path):
            split_path = self.dataset_path  # Try using the main path if split subdirectory doesn't exist
        
        # Get all image files with supported extensions
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        self.image_files = []
        
        for root, _, files in os.walk(split_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    self.image_files.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.image_files.sort()
        
        # Load captions if available (plain text files with same name as images)
        self.captions = []
        for img_path in self.image_files:
            caption_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                except:
                    caption = ""
            else:
                caption = ""
            self.captions.append(caption)
    
    def _load_from_huggingface(self):
        """Load dataset from HuggingFace datasets"""
        try:
            from datasets import load_dataset
            
            # Load dataset
            dataset_name = getattr(self.config, 'hf_dataset_name', None)
            if dataset_name is None:
                raise ValueError("HuggingFace dataset name not specified in config")
                
            # Load dataset split
            dataset = load_dataset(dataset_name, split=self.hf_split)
            
            # Get image and caption columns
            image_column = getattr(self.config, 'hf_image_column', 'image')
            caption_column = getattr(self.config, 'hf_caption_column', 'caption')
            
            # Extract images and captions
            for item in dataset:
                # For HF datasets, we'll store the dataset index instead of file path
                self.image_files.append(item[image_column])
                
                if caption_column in item:
                    self.captions.append(item[caption_column])
                else:
                    self.captions.append("")
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise
    
    def _init_shared_clusters(self, cluster_labels):
        """
        Initialize cluster assignments from ClusterManager
        
        Args:
            cluster_labels: Pre-computed cluster labels
        """
        # Check input
        if cluster_labels is None:
            self.logger.warning("Received None for cluster_labels")
            # Initialize with -1 (unassigned)
            self.cluster_assignments = np.full(len(self.image_files), -1)
            return
        
        # Check length match
        if len(cluster_labels) != len(self.image_files):
            self.logger.warning(f"Cluster labels length ({len(cluster_labels)}) does not match dataset length ({len(self.image_files)})")
            
            # Create placeholder for actual dataset length
            self.cluster_assignments = np.full(len(self.image_files), -1)
            
            # Copy available labels
            if len(cluster_labels) < len(self.image_files):
                # Fewer labels than images, copy what we have and leave the rest unassigned
                self.cluster_assignments[:len(cluster_labels)] = cluster_labels
                self.logger.warning(f"Only {len(cluster_labels)}/{len(self.image_files)} samples have cluster assignments")
            else:
                # More labels than images, truncate
                self.cluster_assignments = cluster_labels[:len(self.image_files)]
                self.logger.warning(f"Truncated {len(cluster_labels) - len(self.image_files)} excess cluster labels")
        else:
            # Equal lengths, perfect match
            self.cluster_assignments = cluster_labels

    def _init_buckets(self):
        """Initialize buckets for aspect ratio batching"""
        # Get bucket specifications from config
        if hasattr(self.config, 'buckets'):
            buckets = self.config.buckets
        else:
            # Default buckets: square, 4:3, 3:4, 16:9, 9:16
            buckets = [
                (256, 256),   # Square
                (288, 224),   # 4:3 landscape
                (224, 288),   # 3:4 portrait
                (320, 192),   # 16:9 landscape
                (192, 320),   # 9:16 portrait
            ]
        
        # Process images and assign to buckets
        bucket_indices = {i: [] for i in range(len(buckets))}
        
        for idx, img_path in enumerate(self.image_files):
            try:
                # Get image dimensions
                if isinstance(img_path, str) and os.path.exists(img_path):
                    # Regular file path
                    with Image.open(img_path) as img:
                        width, height = img.size
                elif hasattr(img_path, 'width') and hasattr(img_path, 'height'):
                    # HuggingFace dataset image
                    width, height = img_path.width, img_path.height
                else:
                    # Unknown format, assume square
                    width, height = 1, 1
                    
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                # Find closest bucket
                bucket_aspects = [w/h for w, h in buckets]
                distances = [abs(aspect_ratio - ba) for ba in bucket_aspects]
                closest_bucket = np.argmin(distances)
                
                # Add to bucket
                bucket_indices[closest_bucket].append(idx)
                
            except Exception as e:
                # Skip problematic images
                self.logger.warning(f"Error processing image {img_path}: {e}")
        
        # Store bucket info
        self.buckets = buckets
        self.bucket_indices = bucket_indices
        
        # Log bucket sizes
        for i, (w, h) in enumerate(buckets):
            count = len(bucket_indices[i])
            self.logger.info(f"Bucket {i} ({w}x{h}): {count} images")
    
    def _init_bucket_assignments(self):
        """Initialize bucket assignments for efficient batching"""
        # Map bucket indices to original indices
        self.bucket_assignments = {}
        for bucket_idx, indices in self.bucket_indices.items():
            for i, idx in enumerate(indices):
                self.bucket_assignments[idx] = (bucket_idx, i)
    
    def __len__(self):
        """Get dataset length"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get dataset item with clustering information"""
        # Get image path
        img_path = self.image_files[idx]
        
        # Load and transform image
        try:
            if isinstance(img_path, str) and os.path.exists(img_path):
                # Regular file path
                img = Image.open(img_path).convert('RGB')
            else:
                # HuggingFace dataset image or other format
                img = img_path.convert('RGB') if hasattr(img_path, 'convert') else img_path
                
            # Get bucket assignment
            if idx in self.bucket_assignments:
                bucket_idx, _ = self.bucket_assignments[idx]
                target_size = self.buckets[bucket_idx]
            else:
                # Default to square if no bucket assignment
                target_size = (256, 256)
                
            # Apply transforms if provided
            if self.transforms is not None:
                img = self.transforms(img, target_size=target_size)
            else:
                # Basic resize if no transforms provided
                img = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                ])(img)
                
        except Exception as e:
            # Return a blank tensor on error
            self.logger.warning(f"Error loading image {img_path}: {e}")
            img = torch.zeros((3, 256, 256), dtype=torch.float32)
        
        # Get caption
        caption = self.captions[idx] if idx < len(self.captions) else ""
        
        # Get cluster label if available
        if self.cluster_assignments is not None and idx < len(self.cluster_assignments):
            cluster = int(self.cluster_assignments[idx])
        else:
            cluster = -1  # -1 indicates no cluster
            
        # Return as dict for flexibility
        return {
            "image": img,
            "caption": caption,
            "cluster": cluster,
            "index": idx,
            "path": img_path if isinstance(img_path, str) else f"sample_{idx}"
        }
        
    def update_cluster_assignments(self, cluster_labels):
        """Update cluster assignments with new labels"""
        self._init_shared_clusters(cluster_labels)
        
    def get_bucket_sampler(self, batch_size=32, shuffle=True, drop_last=True):
        """
        Create a batched sampler that maintains aspect ratio grouping
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            BucketBatchSampler for efficient batching
        """
        from torch.utils.data import Sampler
        import math
        
        class BucketBatchSampler(Sampler):
            def __init__(self, bucket_indices, batch_size, shuffle=True, drop_last=True):
                self.bucket_indices = bucket_indices
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.drop_last = drop_last
                
                # Calculate number of batches per bucket
                self.batch_counts = {}
                self.total_batches = 0
                
                for bucket, indices in self.bucket_indices.items():
                    if drop_last:
                        count = len(indices) // batch_size
                    else:
                        count = math.ceil(len(indices) / batch_size)
                    self.batch_counts[bucket] = count
                    self.total_batches += count
            
            def __iter__(self):
                # Create batches for each bucket
                bucket_batches = {}
                
                for bucket, indices in self.bucket_indices.items():
                    # Copy indices to avoid modifying original
                    bucket_indices = indices.copy()
                    
                    # Shuffle if needed
                    if self.shuffle:
                        random.shuffle(bucket_indices)
                    
                    # Create batches
                    bucket_batches[bucket] = []
                    for i in range(0, len(bucket_indices), self.batch_size):
                        if i + self.batch_size <= len(bucket_indices) or not self.drop_last:
                            batch = bucket_indices[i:i + self.batch_size]
                            if len(batch) > 0:  # Only add non-empty batches
                                bucket_batches[bucket].append(batch)
                
                # Determine batch order
                bucket_order = list(self.bucket_indices.keys())
                if self.shuffle:
                    random.shuffle(bucket_order)
                
                # Interleave batches from different buckets
                batch_list = []
                
                # Continue until all batches are used
                while any(len(bucket_batches[b]) > 0 for b in bucket_order):
                    for bucket in bucket_order:
                        if bucket_batches[bucket]:
                            batch_list.append(bucket_batches[bucket].pop(0))
                
                # Yield batches
                for batch in batch_list:
                    yield batch
            
            def __len__(self):
                return self.total_batches
                
        return BucketBatchSampler(self.bucket_indices, batch_size, shuffle, drop_last)

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