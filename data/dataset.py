"""Dataset classes for Decentralized Diffusion Models."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class DDMDataset(Dataset):
    """Dataset with cluster assignments and multi-resolution buckets for DDM training"""
    def __init__(self, root_dir, transform=None, cluster_labels=None, include_metadata=True):
        self.root = root_dir
        self.transform = transform
        self.include_metadata = include_metadata
        
        # Validate and filter image files
        logger.info("Initializing DDMDataset and validating images...")
        self.image_files = self._validate_files()
        logger.info(f"DDMDataset initialized with {len(self.image_files)} valid images")
        
        # Extract captions from filenames
        self.captions = {}
        for img_file in self.image_files:
            # Extract caption from filename (assuming format: caption_hash.ext)
            base_name = os.path.splitext(img_file)[0]
            caption = base_name.split('_')[0].replace('-', ' ')
            self.captions[img_file] = caption
        
        # Collect actual image dimensions
        self.actual_sizes = self._collect_image_sizes()
        
        # Dynamic bucket generation
        self.buckets = self._generate_dynamic_buckets()
        
        # Initialize cluster labels
        self.cluster_labels = np.array(cluster_labels) if cluster_labels is not None else np.zeros(len(self))
        self.image_buckets = [None] * len(self)
        self.bucket_indices = defaultdict(list)
        self.assign_buckets()

    def _validate_files(self):
        """Validate all image files and return only the valid ones"""
        valid_files = []
        invalid_count = 0
        
        for fname in tqdm(os.listdir(self.root), desc="Validating images"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if self._is_valid_image(os.path.join(self.root, fname)):
                    valid_files.append(fname)
                else:
                    invalid_count += 1
                    
        logger.info(f"Found {invalid_count} invalid images that will be excluded")
        return valid_files

    def _is_valid_image(self, path):
        """Check if an image file is valid by attempting to open it"""
        try:
            with Image.open(path) as img:
                # Just check if we can open it and access basic properties
                img.size
                return True
        except Exception as e:
            logger.debug(f"Invalid image {path}: {str(e)}")
            return False

    def _collect_image_sizes(self):
        """Collect all image dimensions in the dataset"""
        sizes = []
        for img_file in tqdm(self.image_files, desc="Collecting sizes"):
            try:
                with Image.open(os.path.join(self.root, img_file)) as img:
                    w, h = img.size
                    sizes.append([w, h])
            except Exception as e:
                # This shouldn't happen since we've already validated the files,
                # but just in case, log it and continue
                logger.warning(f"Error getting size for {img_file}: {e}")
                continue
        return np.array(sizes)

    def _generate_dynamic_buckets(self, num_buckets=20):
        """Generate buckets based on actual image size distribution"""
        if len(self.actual_sizes) == 0:
            return [(512, 512)]
            
        # Cluster similar sizes using K-means
        kmeans = MiniBatchKMeans(n_clusters=min(num_buckets, len(self.actual_sizes)))
        kmeans.fit(self.actual_sizes)
        
        # Get cluster centers and round to nearest 256 (VAE downscale * patch size)
        buckets = []
        for center in kmeans.cluster_centers_:
            w = int(round(center[0] / 256)) * 256
            h = int(round(center[1] / 256)) * 256
            buckets.append((max(w, 256), max(h, 256)))  # Ensure minimum size of 256x256
            
        # Deduplicate and return
        return list(set(buckets))

    def find_closest_bucket(self, width, height):
        """Find closest dynamic bucket using aspect ratio aware distance"""
        target_aspect = width / height
        min_distance = float('inf')
        closest = None
        
        for bucket in self.buckets:
            bw, bh = bucket
            bucket_aspect = bw / bh
            
            # Calculate distance with aspect ratio priority
            aspect_diff = abs(bucket_aspect - target_aspect) * 1000  # Higher weight for aspect
            size_diff = abs(bw - width) + abs(bh - height)
            total_diff = aspect_diff + size_diff
            
            if total_diff < min_distance:
                min_distance = total_diff
                closest = bucket
                
        return closest or (512, 512)

    def assign_buckets(self):
        """Assign each image to its closest resolution bucket"""
        for idx, img_file in enumerate(tqdm(self.image_files, desc="Assigning buckets")):
            img_path = os.path.join(self.root, img_file)
            try:
                # Get image size
                with Image.open(img_path) as img:
                    w, h = img.size
                
                # Find closest bucket
                closest_bucket = self.find_closest_bucket(w, h)
                
                # Assign bucket
                self.image_buckets[idx] = closest_bucket
                self.bucket_indices[closest_bucket].append(idx)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                # Assign a default bucket
                self.image_buckets[idx] = self.buckets[0]
                self.bucket_indices[self.buckets[0]].append(idx)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            
            # Get assigned bucket
            bucket = self.image_buckets[idx]
            target_w, target_h = bucket
            
            # Create specific transform for this resolution
            bucket_transform = T.Compose([
                T.Resize((target_h, target_w)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Get caption for this image
            caption = self.captions[self.image_files[idx]]
            
            return {
                "image": bucket_transform(image),
                "cluster": self.cluster_labels[idx],
                "caption": caption
            }
            
        except Exception as e:
            logger.error(f"Error loading {self.image_files[idx]}: {e}")
            # Instead of returning a placeholder, raise the exception to be handled properly
            raise

class FeatureDataset(Dataset):
    """Dedicated dataset for feature extraction without bucket logic"""
    def __init__(self, root_dir, config=None):
        from config import DDMConfig
        
        self.root = root_dir
        self.config = config or DDMConfig()  # Use provided config or default
        
        # Include webp files in feature extraction but validate them first
        self.image_files = self._validate_files()
        self.dino_size = self.config.dino_size
        self.transform = T.Compose([
            T.Resize(self.dino_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.dino_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info(f"FeatureDataset initialized with {len(self.image_files)} valid images")

    def _validate_files(self):
        """Validate all image files and return only the valid ones"""
        valid_files = []
        invalid_count = 0
        
        for fname in tqdm(os.listdir(self.root), desc="Validating images"):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if self._is_valid_image(os.path.join(self.root, fname)):
                    valid_files.append(fname)
                else:
                    invalid_count += 1
                    
        logger.info(f"Found {invalid_count} invalid images that will be excluded")
        return valid_files

    def _is_valid_image(self, path):
        """Check if an image file is valid by attempting to open it"""
        try:
            with Image.open(path) as img:
                # Just check if we can open it and access basic properties
                img.size
                return True
        except Exception as e:
            logger.debug(f"Invalid image {path}: {str(e)}")
            return False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image instead of failing
            blank = torch.zeros(3, self.dino_size, self.dino_size)
            return blank

class BucketBatchSampler:
    """
    Sampler that creates batches of samples with the same bucket
    to avoid excessive padding and maintain efficiency
    """
    def __init__(self, bucket_indices, batch_size, shuffle=True):
        self.bucket_indices = bucket_indices  # Dict mapping bucket -> list of sample indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Flatten bucket indices into (bucket_id, sample_id) pairs
        self.flattened = []
        for bucket_id, indices in self.bucket_indices.items():
            self.flattened.extend([(bucket_id, sample_id) for sample_id in indices])
    
    def __iter__(self):
        # Shuffle all samples if required
        if self.shuffle:
            np.random.shuffle(self.flattened)
        
        # Group by bucket and yield batches
        batches = []
        current_bucket = None
        current_batch = []
        
        for bucket_id, sample_id in self.flattened:
            # If we're in a new bucket or the batch is full, yield the batch
            if current_bucket != bucket_id or len(current_batch) >= self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [sample_id]
                current_bucket = bucket_id
            else:
                current_batch.append(sample_id)
                
            # If batch is full, add it to batches
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)
            
        # Shuffle batches to mix different buckets
        if self.shuffle:
            np.random.shuffle(batches)
            
        # Return iterator over batches
        return iter(batches)
        
    def __len__(self):
        return len(self.flattened) // self.batch_size + (0 if len(self.flattened) % self.batch_size == 0 else 1)

def create_expert_bucket_loaders(dataset, config, world_size=1, rank=0):
    """Create data loaders for each expert with multi-resolution bucket batching"""
    # Ensure dataset and cluster_labels have the same length
    if len(dataset.cluster_labels) != len(dataset):
        logger.warning(f"Cluster labels length ({len(dataset.cluster_labels)}) doesn't match dataset length ({len(dataset)})")
        # Resize cluster_labels to match dataset length
        if len(dataset.cluster_labels) < len(dataset):
            # Extend with zeros
            dataset.cluster_labels = np.pad(dataset.cluster_labels, 
                                           (0, len(dataset) - len(dataset.cluster_labels)),
                                           'constant')
        else:
            # Truncate
            dataset.cluster_labels = dataset.cluster_labels[:len(dataset)]
    
    # Group samples by cluster
    cluster_groups = defaultdict(list)
    for idx in range(len(dataset)):
        cluster = int(dataset.cluster_labels[idx]) % config.num_experts  # Ensure valid cluster index
        cluster_groups[cluster].append(idx)
    
    # Ensure all experts have data by redistributing if needed
    for expert_idx in range(config.num_experts):
        if expert_idx not in cluster_groups or len(cluster_groups[expert_idx]) < config.batch_size:
            logger.warning(f"Expert {expert_idx} has insufficient samples, redistributing data")
            
            # Find experts with most samples to redistribute from
            donor_experts = sorted([(k, len(v)) for k, v in cluster_groups.items() if k != expert_idx and len(v) > config.batch_size*2],
                                  key=lambda x: x[1], reverse=True)
            
            # Create or extend the group for this expert
            if expert_idx not in cluster_groups:
                cluster_groups[expert_idx] = []
            
            # Take samples from donors until we have enough
            samples_needed = max(config.batch_size*2 - len(cluster_groups[expert_idx]), 0)
            
            for donor_idx, donor_size in donor_experts:
                # Calculate how many samples to take
                samples_to_take = min(donor_size // 3, samples_needed)  # Take up to 1/3 of donor's samples
                if samples_to_take <= 0:
                    continue
                    
                # Take samples from donor
                donor_samples = cluster_groups[donor_idx][-samples_to_take:]
                cluster_groups[donor_idx] = cluster_groups[donor_idx][:-samples_to_take]
                
                # Add to current expert and update cluster labels
                cluster_groups[expert_idx].extend(donor_samples)
                for idx in donor_samples:
                    dataset.cluster_labels[idx] = expert_idx
                    
                samples_needed -= samples_to_take
                if samples_needed <= 0:
                    break
            
            # If we still don't have enough, log a warning but continue
            if len(cluster_groups[expert_idx]) < config.batch_size:
                logger.warning(f"Expert {expert_idx} still has only {len(cluster_groups[expert_idx])} samples after redistribution")
    
    # Create a loader for each expert
    expert_loaders = []
    for expert_idx in range(config.num_experts):
        indices = cluster_groups[expert_idx]
        logger.info(f"Expert {expert_idx} has {len(indices)} training samples")
        
        expert_dataset = Subset(dataset, indices)
        
        # Get bucket groups for this expert
        expert_bucket_indices = defaultdict(list)
        for i, idx in enumerate(indices):
            bucket = dataset.image_buckets[idx]
            expert_bucket_indices[bucket].append(i)
        
        # Create bucket batch sampler with appropriate batch size
        batch_size = config.expert_batch_size
        sampler = BucketBatchSampler(
            expert_bucket_indices,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create loader with bucket batch sampler
        loader = DataLoader(
            expert_dataset,
            batch_sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        expert_loaders.append(loader)
    
    return expert_loaders 