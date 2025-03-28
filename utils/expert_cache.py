"""Expert Cache Manager for memory-efficient handling of multiple expert models."""

import torch
import logging
import threading
from collections import OrderedDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from utils.fsdp import wrap_model_with_fsdp, configure_optimizer_for_fsdp
import torch.distributed as dist
from bitsandbytes.optim import AdamW8bit

class ExpertCacheManager:
    """
    Simplified expert model manager for memory-efficient handling of multiple experts.
    Uses LRU caching strategy with optional CPU offloading.
    """
    
    def __init__(self, config, device="cuda", max_experts=None, cpu_offload=None, logger=None):
        """
        Initialize expert cache manager
        
        Args:
            config: Configuration object with expert caching settings
            device: Device to load experts on
            max_experts: Maximum number of experts to keep in memory
            cpu_offload: Whether to offload experts to CPU
            logger: Centralized logger instance
        """
        self.config = config
        self.device = device
        self.max_experts = max_experts or getattr(config, 'max_experts_in_memory', 3)
        self.cpu_offload = cpu_offload if cpu_offload is not None else getattr(config, 'expert_offload_to_cpu', True)
        self.rank = dist.get_rank()  # Add rank attribute
        
        # Use the passed logger or create a fallback
        self.logger = logger if logger else logging.getLogger(f"ExpertCacheManager_{self.rank}_fallback")
        
        # Simple LRU cache using OrderedDict
        self.expert_cache = OrderedDict()  # Maps expert_idx -> expert model
        self.cpu_cache = {}  # Maps expert_idx -> CPU copy of expert model
        
        # Lock for thread-safe operations
        self.cache_lock = threading.RLock()
        
        # Use self.logger for logging
        self.logger.info(f"Initialized simplified ExpertCacheManager with max_experts={self.max_experts}, cpu_offload={self.cpu_offload}")
    
    def get_expert(self, expert_idx, expert_factory_fn):
        """Get expert from cache, loading if necessary"""
        with self.cache_lock:
            try:
                if expert_idx in self.expert_cache:
                    expert = self.expert_cache[expert_idx]
                    self.expert_cache.move_to_end(expert_idx)
                    # Use self.logger
                    # self.logger.debug(f"Expert {expert_idx} found in GPU cache.")
                    return expert
                
                if expert_idx in self.cpu_cache:
                    expert = self.cpu_cache.pop(expert_idx)
                    # Use self.logger
                    self.logger.debug(f"Moving expert {expert_idx} from CPU to {self.device}")
                    if isinstance(expert, FSDP):
                        expert.to(self.device)
                    else:
                        expert = expert.to(self.device)
                    self._add_to_cache(expert_idx, expert)
                    return expert

                # Use self.logger
                self.logger.info(f"Creating new expert {expert_idx}...")
                # Create new expert trainer
                expert = expert_factory_fn(expert_idx)
                
                # Only wrap the expert model with FSDP if not already wrapped
                # Pass self.logger to sub-functions if they need it
                if hasattr(expert, 'expert') and not isinstance(expert.expert, FSDP):
                    expert.expert = wrap_model_with_fsdp(
                        expert.expert,
                        self.config,
                        param_init_fn=lambda m: m.to_empty(device=self.device, recurse=False),
                        rank=self.rank
                    )
                    
                    # Reconfigure optimizer for FSDP
                    # Pass self.logger if configure_optimizer_for_fsdp needs it
                    expert.optimizer = configure_optimizer_for_fsdp(
                        expert.expert,
                        AdamW8bit,
                        lr=self.config.learning_rate,
                        betas=self.config.adam_betas,
                        weight_decay=self.config.weight_decay
                    )
                    
                    # Initialize scaler for mixed precision training
                    expert.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_mixed_precision)
                
                if not hasattr(expert, 'router'):
                    # Assign router if needed (assuming it's passed or accessible)
                    # This logic might need refinement based on how router is provided
                    # self.logger.warning(f"Expert {expert_idx} missing router reference during cache creation.")
                    # Example: expert.router = some_accessible_router
                    pass # Placeholder if router isn't strictly needed here
                
                self._add_to_cache(expert_idx, expert)
                return expert

            except Exception as e:
                # Use self.logger
                self.logger.error(f"Critical error loading expert {expert_idx}: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to load expert {expert_idx}") from e
    
    def _add_to_cache(self, expert_idx, expert):
        """Add expert to cache, evicting if necessary"""
        # If cache is full, evict least recently used expert
        if len(self.expert_cache) >= self.max_experts:
            # Get the first item (least recently used)
            lru_idx, lru_expert = next(iter(self.expert_cache.items()))
            
            # Use self.logger
            self.logger.debug(f"Cache full ({len(self.expert_cache)} >= {self.max_experts}). Evicting expert {lru_idx}.")
            
            # Move to CPU if offloading is enabled
            if self.cpu_offload:
                # Use self.logger
                self.logger.debug(f"Moving evicted expert {lru_idx} from GPU to CPU")
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
            # Optional: Explicitly delete if memory is tight, though Python's GC should handle it
            # del lru_expert

        # Add new expert to cache
        self.expert_cache[expert_idx] = expert
        # Use self.logger
        self.logger.debug(f"Added expert {expert_idx} to GPU cache. Cache size: {len(self.expert_cache)}.")
        
    def clear_cache(self):
        """Clear all caches"""
        with self.cache_lock:
            # Use self.logger
            self.logger.info("Clearing expert cache (GPU and CPU).")
            self.expert_cache.clear()
            self.cpu_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def prefetch_expert(self, expert_idx, expert_builder_fn):
        """
        Prefetch expert to CPU (simplified non-threaded version)
        
        Args:
            expert_idx: Expert index
            expert_builder_fn: Function to build expert
        """
        if not self.cpu_offload:
            # Use self.logger
            self.logger.debug("Prefetching skipped: CPU offload disabled.")
            return
            
        with self.cache_lock:
            # Skip if already cached
            if expert_idx in self.cpu_cache or expert_idx in self.expert_cache:
                # Use self.logger
                self.logger.debug(f"Prefetch skipped: Expert {expert_idx} already cached.")
                return
                
            # Build expert on CPU
            # Use self.logger
            self.logger.debug(f"Prefetching expert {expert_idx} to CPU")
            try:
                 with torch.no_grad():
                     # Ensure builder fn creates model appropriately (potentially on CPU directly)
                     expert = expert_builder_fn(expert_idx)
                     # Explicitly move to CPU if not already there
                     if hasattr(expert, 'to'):
                          self.cpu_cache[expert_idx] = expert.to('cpu')
                     else: # Handle cases where factory returns non-module objects?
                          self.cpu_cache[expert_idx] = expert
            except Exception as e:
                 self.logger.error(f"Error prefetching expert {expert_idx}: {e}", exc_info=True)

    def shutdown(self):
        """
        Clean up resources used by the cache manager
        """
        try:
            # Clear references to experts
            if hasattr(self, 'expert_cache'):
                self.expert_cache.clear()
            if hasattr(self, 'cpu_cache'):
                self.cpu_cache.clear()
            
            # Free memory
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Use self.logger
            self.logger.info("Expert cache manager shutdown completed successfully")
        except Exception as e:
            # Use self.logger
            self.logger.error(f"Error during expert cache manager shutdown: {str(e)}")
            return False
        return True 

    def _enforce_cache_limits(self):
        """Enforce cache size limits, offloading experts if necessary"""
        while len(self.expert_cache) > self.max_experts: # Use while loop to handle multiple evictions if needed
            # Get the first expert (least recently used)
            try:
                lru_idx, lru_expert = next(iter(self.expert_cache.items()))
            except StopIteration:
                break # Cache is empty

            # Use self.logger
            self.logger.warning(f"Cache limit ({self.max_experts}) exceeded. Offloading LRU expert {lru_idx}.")

            # Move to CPU if offloading is enabled
            if self.cpu_offload:
                # Use self.logger
                self.logger.debug(f"Offloading expert {lru_idx} to CPU due to cache limit")
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
            # Optional: del lru_expert and gc.collect() if needed 