"""Expert Cache Manager for memory-efficient handling of multiple expert models."""

import torch
import logging
import threading
from collections import OrderedDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
    
    def get_expert(self, expert_idx, expert_factory_fn=None):
        """Get expert from cache, loading if necessary"""
        with self.cache_lock:  # Always use the lock for thread safety
            try:
                # Use expert_cache instead of active_experts
                if expert_idx in self.expert_cache:
                    # Reorder expert_cache to make this expert the most recently used
                    expert = self.expert_cache.pop(expert_idx)
                    self.expert_cache[expert_idx] = expert
                    return expert
                
                # Use cpu_cache instead of offloaded_experts
                if expert_idx in self.cpu_cache:
                    # Load from CPU back to GPU
                    expert = self.cpu_cache[expert_idx]
                    # Move to device properly - FSDP-aware
                    if isinstance(expert, FSDP):
                        # For FSDP models, ensure proper device placement
                        if expert.device != self.device:
                            raise RuntimeError(f"FSDP expert {expert_idx} on wrong device {expert.device}")
                    else:
                        # Regular model
                        expert = expert.to(self.device)
                    
                    # Add to active cache
                    self.expert_cache[expert_idx] = expert
                    # Remove from CPU cache
                    del self.cpu_cache[expert_idx]
                    return expert
                
                # Not in any cache, create new expert
                if expert_factory_fn:
                    expert = expert_factory_fn(expert_idx)
                    # Add to active cache
                    self.expert_cache[expert_idx] = expert
                    # Check if we need to evict experts
                    self._enforce_cache_limits()
                    return expert
                
                # No factory function and expert not found
                return None
            
            except Exception as e:
                logger.error(f"Error getting expert {expert_idx} from cache: {str(e)}")
                # Attempt to recreate on error
                if expert_factory_fn:
                    try:
                        expert = expert_factory_fn(expert_idx)
                        self.expert_cache[expert_idx] = expert
                        return expert
                    except Exception as inner_e:
                        logger.error(f"Failed to recreate expert {expert_idx}: {str(inner_e)}")
                return None
    
    def _add_to_cache(self, expert_idx, expert):
        """Add expert to cache, evicting if necessary"""
        # If cache is full, evict least recently used expert
        if len(self.expert_cache) >= self.max_experts:
            # Get the first item (least recently used)
            lru_idx, lru_expert = next(iter(self.expert_cache.items()))
            
            # Move to CPU if offloading is enabled
            if self.cpu_offload:
                logger.debug(f"Moving expert {lru_idx} from GPU to CPU")
                if isinstance(lru_expert, FSDP):
                    # Properly handle FSDP model movement
                    with torch.no_grad():
                        lru_expert.to('cpu')
                    self.cpu_cache[lru_idx] = lru_expert
                else:
                    # For non-FSDP models, simple CPU transfer
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

    def shutdown(self):
        """
        Clean up resources used by the cache manager
        """
        try:
            # Clear references to experts
            if hasattr(self, 'expert_cache'):
                self.expert_cache.clear()
            
            # Free memory
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Expert cache manager shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during expert cache manager shutdown: {str(e)}")
            return False
        return True 

    def _enforce_cache_limits(self):
        """Enforce cache size limits, offloading experts if necessary"""
        if len(self.expert_cache) > self.max_experts:
            # Get the first expert (least recently used)
            lru_idx, lru_expert = next(iter(self.expert_cache.items()))
            
            # Move to CPU if offloading is enabled
            if self.cpu_offload:
                logger.debug(f"Offloading expert {lru_idx} to CPU due to cache limit")
                # For FSDP models, handle differently
                if isinstance(lru_expert, FSDP):
                    # Properly handle FSDP model movement
                    with torch.no_grad():
                        lru_expert.to('cpu')
                    self.cpu_cache[lru_idx] = lru_expert
                else:
                    # For non-FSDP models, simple CPU transfer
                    self.cpu_cache[lru_idx] = lru_expert.to('cpu')
            
            # Remove from GPU cache
            del self.expert_cache[lru_idx] 