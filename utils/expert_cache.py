"""Expert Cache Manager for memory-efficient handling of multiple expert models."""

import torch
import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ExpertCacheManager:
    """
    Simplified expert model manager for memory-efficient handling of multiple experts.
    Uses LRU caching strategy with optional CPU offloading.
    """
    
    def __init__(self, config, device="cuda", max_experts=None, cpu_offload=None):
        """
        Initialize expert cache manager
        
        Args:
            config: Configuration object with expert caching settings
            device: Device to load experts on
            max_experts: Maximum number of experts to keep in memory
            cpu_offload: Whether to offload experts to CPU
        """
        self.config = config
        self.device = device
        self.max_experts = max_experts or getattr(config, 'max_experts_in_memory', 3)
        self.cpu_offload = cpu_offload if cpu_offload is not None else getattr(config, 'expert_offload_to_cpu', True)
        
        # Simple LRU cache using OrderedDict
        self.expert_cache = OrderedDict()  # Maps expert_idx -> expert model
        self.cpu_cache = {}  # Maps expert_idx -> CPU copy of expert model
        
        # Lock for thread-safe operations
        self.cache_lock = threading.RLock()
        
        logger.info(f"Initialized simplified ExpertCacheManager with max_experts={self.max_experts}, cpu_offload={self.cpu_offload}")
    
    def get_expert(self, expert_idx, expert_builder_fn):
        """
        Get an expert model, loading it if necessary
        
        Args:
            expert_idx: Expert index
            expert_builder_fn: Function to build expert if not cached
            
        Returns:
            Expert model on target device
        """
        with self.cache_lock:
            # Check if expert is already in GPU cache
            if expert_idx in self.expert_cache:
                # Move expert to end of LRU order
                expert = self.expert_cache.pop(expert_idx)
                self.expert_cache[expert_idx] = expert
                return expert
                
            # Check if expert is in CPU cache
            if self.cpu_offload and expert_idx in self.cpu_cache:
                # Load from CPU to GPU
                logger.debug(f"Moving expert {expert_idx} from CPU to GPU")
                expert = self.cpu_cache.pop(expert_idx)
                expert = expert.to(self.device)
                
                # Add to GPU cache and manage cache size
                self._add_to_cache(expert_idx, expert)
                return expert
            
            # Expert not in cache, create it
            logger.debug(f"Creating new expert {expert_idx}")
            expert = expert_builder_fn(expert_idx)
            expert = expert.to(self.device)
            
            # Add to cache and manage cache size
            self._add_to_cache(expert_idx, expert)
            return expert
    
    def _add_to_cache(self, expert_idx, expert):
        """Add expert to cache, evicting if necessary"""
        # If cache is full, evict least recently used expert
        if len(self.expert_cache) >= self.max_experts:
            # Get the first item (least recently used)
            lru_idx, lru_expert = next(iter(self.expert_cache.items()))
            
            # Move to CPU if offloading is enabled
            if self.cpu_offload:
                logger.debug(f"Moving expert {lru_idx} from GPU to CPU")
                self.cpu_cache[lru_idx] = lru_expert.to('cpu')
            
            # Remove from GPU cache
            self.expert_cache.pop(lru_idx)
            
        # Add new expert to cache
        self.expert_cache[expert_idx] = expert
        
    def clear_cache(self):
        """Clear all caches"""
        with self.cache_lock:
            self.expert_cache.clear()
            self.cpu_cache.clear()
            torch.cuda.empty_cache()
            
    def prefetch_expert(self, expert_idx, expert_builder_fn):
        """
        Prefetch expert to CPU (simplified non-threaded version)
        
        Args:
            expert_idx: Expert index
            expert_builder_fn: Function to build expert
        """
        if not self.cpu_offload:
            return  # No prefetching without CPU offload
            
        with self.cache_lock:
            # Skip if already cached
            if expert_idx in self.cpu_cache or expert_idx in self.expert_cache:
                return
                
            # Build expert on CPU
            logger.debug(f"Prefetching expert {expert_idx} to CPU")
            with torch.no_grad():
                expert = expert_builder_fn(expert_idx)
                self.cpu_cache[expert_idx] = expert.to('cpu') 