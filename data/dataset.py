"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL  # Add direct import of PIL module
from collections import defaultdict
import logging
import time
import glob
import json

import random
import io
import torchvision.transforms as transforms

# Import centralized utilities
from utils.distributed import is_main_process, get_rank, broadcast_object
from utils.logging import setup_logger, setup_distributed_logger
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
            
            total_files = len(all_files)
            logger.info(f"Found {total_files} potential image files to validate")
            last_update_time = time.time()
            update_interval = 2.0  # Update progress every 2 seconds
            
            # Check validity of each file
            for i, img_path in enumerate(all_files):
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
                
                # Show progress periodically
                current_time = time.time()
                if current_time - last_update_time > update_interval or i == total_files - 1:
                    progress = (i + 1) / total_files * 100
                    elapsed = current_time - start_time
                    remaining = elapsed / (i + 1) * (total_files - i - 1) if i > 0 else 0
                    
                    logger.info(f"Validation progress: {progress:.1f}% ({i+1}/{total_files}) - "
                               f"Valid: {len(valid_files)}, Invalid: {invalid_count} - "
                               f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                    last_update_time = current_time
            
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
        
        batch_size = len(file_paths)
        logger.debug(f"Validating batch of {batch_size} images with {workers} workers")
        start_time = time.time()
        
        # Process images in parallel
        valid_files = []
        invalid_files = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            validator_fn = partial(_validate_single, min_size=min_size)
            
            # Process files with progress tracking
            processed = 0
            last_update_time = time.time()
            update_interval = 5.0  # Update every 5 seconds for large batches
            
            for is_valid, img_path in executor.map(validator_fn, file_paths):
                if is_valid:
                    valid_files.append(img_path)
                    # Add to cache
                    cls._valid_files_cache[img_path] = True
                else:
                    invalid_files.append(img_path)
                
                # Update progress for large batches
                processed += 1
                current_time = time.time()
                if batch_size > 100 and (current_time - last_update_time > update_interval or processed == batch_size):
                    progress = processed / batch_size * 100
                    elapsed = current_time - start_time
                    remaining = (elapsed / processed) * (batch_size - processed) if processed > 0 else 0
                    speed = processed / elapsed if elapsed > 0 else 0
                    
                    logger.debug(f"Batch validation progress: {progress:.1f}% ({processed}/{batch_size}) - "
                                f"Speed: {speed:.1f} images/sec - "
                                f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
                    last_update_time = current_time
        
        # Final timing
        elapsed = time.time() - start_time
        if batch_size > 10:  # Only log timing for non-trivial batches
            speed = batch_size / elapsed if elapsed > 0 else 0
            logger.debug(f"Batch validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid - "
                        f"Took {elapsed:.2f}s, Speed: {speed:.1f} images/sec")
                    
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
            
        total_files = len(all_files)
        logger.info(f"Found {total_files} potential image files to validate")
        
        # Process in batches
        all_valid_files = []
        total_processed = 0
        batch_start_time = time.time()
        
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            current_batch_size = len(batch)
            
            # Log batch start
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} "
                       f"({i}-{min(i+batch_size, total_files)})")
            
            # Process batch
            batch_start = time.time()
            valid_batch, invalid_batch = cls.validate_images_batch(batch, min_size=min_size)
            batch_time = time.time() - batch_start
            
            all_valid_files.extend(valid_batch)
            total_processed += len(batch)
            
            # Calculate progress metrics
            progress = total_processed / total_files * 100
            elapsed = time.time() - start_time
            avg_time_per_file = elapsed / total_processed if total_processed > 0 else 0
            remaining = avg_time_per_file * (total_files - total_processed)
            
            # Log detailed progress
            logger.info(f"Batch completed in {batch_time:.2f}s - "
                       f"Speed: {current_batch_size/batch_time:.1f} images/sec - "
                       f"Valid: {len(valid_batch)}, Invalid: {len(invalid_batch)}")
            
            logger.info(f"Overall progress: {progress:.1f}% ({total_processed}/{total_files}) - "
                       f"Valid so far: {len(all_valid_files)} - "
                       f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                
        # Final timing
        elapsed = time.time() - start_time
        invalid_count = total_files - len(all_valid_files)
        avg_speed = total_files / elapsed if elapsed > 0 else 0
        
        logger.info(f"Image validation completed in {elapsed:.2f}s: "
                   f"{len(all_valid_files)} valid, {invalid_count} invalid - "
                   f"Average speed: {avg_speed:.1f} images/sec")
                
        # Synchronize after validation
        synchronize()
        
        # Broadcast to other processes
        return broadcast_object(all_valid_files, src=0)

class DDMDataset(Dataset):
    """Implementation with uniform GPU-based splitting"""
    
    def __init__(self, config, split='train', transforms=None, hf_split=None):
        """
        Initialize dataset with uniform GPU-based distribution
        
        Args:
            config: Config object with dataset parameters
            split: Dataset split ('train', 'val', etc.)
            transforms: Image transformations to apply
            hf_split: HuggingFace dataset split name if different from 'split'
        """
        self.config = config
        self.split = split
        self.hf_split = hf_split or split
        self.transforms = transforms
        self.logger = logging.getLogger(__name__)
        
        # Initialize dataset path and storage
        self.dataset_path = getattr(config, 'dataset_path', None)
        self.image_files = []
        self.captions = []
        
        # Initialize GPU-based distribution parameters
        self.num_experts = config.num_experts
        self.expert_assignments = np.zeros(len(self.image_files), dtype=np.int32)
        
        # Initialize buckets and load dataset
        self._load_dataset()
        self._init_buckets()
        self._init_bucket_assignments()
        
        # Uniformly distribute samples across experts
        self._distribute_samples()

    def _distribute_samples(self):
        """Uniformly distribute samples across experts using modulo operation"""
        self.expert_assignments = np.array(
            [i % self.num_experts for i in range(len(self.image_files))],
            dtype=np.int32
        )
        self.logger.info(f"Uniformly distributed {len(self.image_files)} samples across {self.num_experts} experts")

    def _load_dataset(self):
        """Load dataset files and captions"""
        # Check if we should completely bypass loading for fast path
        if getattr(self.config, 'skip_clustering', False):
            self.logger.info("Fast path: Skipping real dataset loading entirely with skip_clustering=True")
            # Return immediately without loading anything
            return
            
        # Continue with normal dataset loading
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
            
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            self.logger.error(f"Dataset path {self.dataset_path} does not exist")
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist")
            
        # Load images recursively for local datasets
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Determine how to load based on dataset path
        if os.path.isdir(self.dataset_path):
            # Local directory - scan for image files recursively
            self.image_files = []
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
            
            # File scan with progress tracking
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        self.image_files.append(os.path.join(root, file))
                        
                # Show progress periodically
                if len(self.image_files) % 100000 == 0 and len(self.image_files) > 0:
                    self.logger.info(f"Found {len(self.image_files)} images so far...")
                    
            # Sort for reproducibility
            self.image_files.sort()
            
            # Load any caption files if they exist
            captions_path = os.path.join(self.dataset_path, 'captions.json')
            if os.path.exists(captions_path):
                self.logger.info(f"Loading captions from {captions_path}")
                try:
                    with open(captions_path, 'r') as f:
                        captions_data = json.load(f)
                        
                    # Format depends on the file structure, adjust as needed
                    if isinstance(captions_data, dict):
                        # Dict mapping file paths to captions
                        self.captions = []
                        for file_path in self.image_files:
                            file_name = os.path.basename(file_path)
                            caption = captions_data.get(file_name, "")
                            self.captions.append(caption)
                    elif isinstance(captions_data, list):
                        # List of captions matching image file order
                        self.captions = captions_data
                        
                    self.logger.info(f"Loaded {len(self.captions)} captions")
                except Exception as e:
                    self.logger.warning(f"Failed to load captions: {e}")
                    
        else:
            # HuggingFace dataset or other format - handle appropriately
            try:
                from datasets import load_dataset
                self.logger.info(f"Loading HuggingFace dataset from {self.dataset_path}")
                
                # Load dataset with appropriate split
                dataset = load_dataset(self.dataset_path, split=self.hf_split)
                
                # Determine image and caption columns
                image_column = getattr(self.config, 'image_column', 'image')
                caption_column = getattr(self.config, 'caption_column', 'caption')
                
                # Store images and captions
                self.image_files = dataset[image_column]
                if caption_column in dataset.column_names:
                    self.captions = dataset[caption_column]
                    
                self.logger.info(f"Loaded {len(self.image_files)} samples from HuggingFace dataset")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {self.dataset_path}: {e}")
                raise
                
        # Log dataset size
        self.logger.info(f"Loaded {len(self.image_files)} images for {self.split} split")
        if len(self.captions) > 0:
            self.logger.info(f"Found captions for {len(self.captions)} images")
    
    def _init_buckets(self):
        """Initialize buckets for aspect ratio batching"""
        # Get bucket specifications from config
        if hasattr(self.config, 'buckets'):
            buckets = self.config.buckets
            self.logger.info(f"Using {len(buckets)} buckets from config: {buckets}")
        else:
            # Default buckets: square, 4:3, 3:4, 16:9, 9:16
            buckets = [
                (256, 256),   # Square
                (288, 224),   # 4:3 landscape
                (224, 288),   # 3:4 portrait
                (320, 192),   # 16:9 landscape
                (192, 320),   # 9:16 portrait
            ]
            self.logger.info(f"Using default buckets: {buckets}")
        
        # FAST PATH: If skip_clustering=True, create simplified bucket assignments
        if getattr(self.config, 'skip_clustering', False):
            # Distribute samples evenly across buckets
            self.logger.info(f"Fast path: Creating uniform bucket assignments")
            
            # Initialize buckets
            self.buckets = buckets
            bucket_indices = {i: [] for i in range(len(buckets))}
            
            # Round-robin assignment to buckets
            for idx in range(len(self.image_files)):
                bucket_idx = idx % len(buckets)
                bucket_indices[bucket_idx].append(idx)
                
            self.bucket_indices = bucket_indices
            
            # Log bucket distribution
            for i, (w, h) in enumerate(buckets):
                count = len(bucket_indices[i])
                percent = count / len(self.image_files) * 100 if len(self.image_files) > 0 else 0
                self.logger.info(f"Bucket {i} ({w}x{h}): {count} images ({percent:.1f}%)")
                
            return
        
        # NORMAL PATH: Process images and assign to buckets based on aspect ratio
        bucket_indices = {i: [] for i in range(len(buckets))}
        
        self.logger.info(f"Assigning {len(self.image_files)} images to {len(buckets)} aspect ratio buckets")
        start_time = time.time()
        total_images = len(self.image_files)
        last_update_time = time.time()
        update_interval = 2.0  # Update progress every 2 seconds
        
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
                
                # Show progress periodically
                current_time = time.time()
                if current_time - last_update_time > update_interval or idx == total_images - 1:
                    progress = (idx + 1) / total_images * 100
                    elapsed = current_time - start_time
                    remaining = elapsed / (idx + 1) * (total_images - idx - 1) if idx > 0 else 0
                    speed = (idx + 1) / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(f"Bucket assignment progress: {progress:.1f}% ({idx+1}/{total_images}) - "
                                   f"Speed: {speed:.1f} images/sec - "
                                   f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
                    last_update_time = current_time
                
            except Exception as e:
                # Skip problematic images
                self.logger.warning(f"Error processing image {img_path}: {e}")
        
        # Store bucket info
        self.buckets = buckets
        self.bucket_indices = bucket_indices
        
        # Log bucket sizes
        bucket_stats = []
        elapsed = time.time() - start_time
        for i, (w, h) in enumerate(buckets):
            count = len(bucket_indices[i])
            percent = count / total_images * 100 if total_images > 0 else 0
            bucket_stats.append(f"Bucket {i} ({w}x{h}): {count} images ({percent:.1f}%)")
            self.logger.info(f"Bucket {i} ({w}x{h}): {count} images ({percent:.1f}%)")
        
        self.logger.info(f"Bucket assignment completed in {elapsed:.2f}s - " 
                       f"Processed {total_images} images at {total_images/elapsed:.1f} images/sec")
    
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
    
    def _default_transform(self, img, bucket_idx=None):
        """Apply default transformations based on bucket dimensions"""
        # Get target dimensions from bucket if provided
        if bucket_idx is not None and 0 <= bucket_idx < len(self.buckets):
            width, height = self.buckets[bucket_idx]
        else:
            # Fallback to default image size
            _, height, width = self.config.image_size
        
        # Resize image to target dimensions
        if isinstance(img, Image.Image):
            # PIL Image
            img = resize_image(img, (width, height))
            img = transforms.ToTensor()(img)
        elif isinstance(img, torch.Tensor):
            # Already a tensor, resize with torch functions
            if img.shape[-2] != height or img.shape[-1] != width:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Normalize
        img = normalize(img)
        return img
        
    def __getitem__(self, idx):
        """Get dataset item by index"""
        try:
            img_path = self.image_files[idx]
            
            # Get bucket assignment
            bucket_idx, position = self.bucket_assignments.get(idx, (0, 0))
            
            # Load image
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                width, height = self.buckets[bucket_idx]
                img = Image.new('RGB', (width, height))
                self.logger.warning(f"Failed to load image {img_path}: {e}")
                
            # Apply transformations
            if self.transforms:
                img = self.transforms(img)
            else:
                img = self._default_transform(img, bucket_idx)
                
            # Get caption if available
            caption = self.captions[idx] if idx < len(self.captions) else ""
            
            return {
                'image': img,
                'caption': caption,
                'path': img_path,
                'expert': self.expert_assignments[idx],
                'bucket': (bucket_idx, position)
            }
        except Exception as e:
            self.logger.error(f"Error processing image at index {idx}: {str(e)}")
            
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
    from utils.logging import setup_distributed_logger
    
    # Initialize logger
    logger = setup_distributed_logger(name="ExpertLoaders", rank=rank)
    
    # Get all indices assigned to each expert
    expert_indices = {}
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        expert_idx = item['expert']
        
        # Skip unassigned samples
        if expert_idx < 0:
            continue
            
        if expert_idx not in expert_indices:
            expert_indices[expert_idx] = []
            
        expert_indices[expert_idx].append(idx)
        
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
    expert_loaders = {}
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