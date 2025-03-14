"""Expert Cache Manager for memory-efficient handling of multiple expert models."""

import torch
import logging
import time
import threading
from collections import OrderedDict
from queue import Queue

logger = logging.getLogger(__name__)

class ExpertCacheManager:
    """
    Manages loading and unloading of expert models to efficiently use GPU memory.
    Implements various caching strategies (LRU, FIFO, RANDOM) with prefetching support.
    """
    
    def __init__(self, config, device="cuda"):
        """
        Initialize expert cache manager
        
        Args:
            config: Configuration object with expert caching settings
            device: Device to load experts on
        """
        self.config = config
        self.device = device
        self.max_experts = getattr(config, 'max_experts_in_memory', 3)
        self.strategy = getattr(config, 'expert_swap_strategy', "LRU")
        self.cpu_offload = getattr(config, 'expert_offload_to_cpu', True)
        self.prefetch_enabled = getattr(config, 'expert_prefetch_next', True)
        
        # Cache and usage tracking
        self.expert_cache = OrderedDict()  # Maps expert_idx -> expert model
        self.cpu_cache = {}  # Maps expert_idx -> CPU copy of expert model
        self.access_count = {}  # Maps expert_idx -> number of accesses
        
        # For LRU strategy
        self.last_used = {}  # Maps expert_idx -> timestamp of last use
        
        # Prefetching
        self.prefetch_queue = Queue()
        self.prefetch_thread = None
        if self.prefetch_enabled:
            self._start_prefetch_thread()
            
        logger.info(f"Initialized ExpertCacheManager with strategy={self.strategy}, max_experts={self.max_experts}")
    
    def _start_prefetch_thread(self):
        """Start background thread for prefetching experts"""
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.info("Started expert prefetching thread")
        
    def _prefetch_worker(self):
        """Worker thread that prefetches experts to CPU or GPU"""
        while True:
            # Get expert to prefetch
            expert_info = self.prefetch_queue.get()
            if expert_info is None:  # Exit signal
                break
                
            expert_idx, expert_builder = expert_info
            
            try:
                # If already in CPU cache, skip
                if expert_idx in self.cpu_cache:
                    continue
                    
                # If already in GPU cache, skip
                if expert_idx in self.expert_cache:
                    continue
                    
                # Build expert and move to CPU
                expert = expert_builder(expert_idx)
                if self.cpu_offload:
                    expert = expert.cpu()
                    self.cpu_cache[expert_idx] = expert
                    logger.debug(f"Prefetched expert {expert_idx} to CPU cache")
                else:
                    # Check if we have room in the GPU cache
                    if len(self.expert_cache) < self.max_experts:
                        self.expert_cache[expert_idx] = expert
                        logger.debug(f"Prefetched expert {expert_idx} directly to GPU cache")
                    else:
                        # No room for prefetching to GPU
                        expert = expert.cpu()
                        self.cpu_cache[expert_idx] = expert
                        logger.debug(f"Prefetched expert {expert_idx} to CPU cache (GPU full)")
            except Exception as e:
                logger.error(f"Error during expert prefetching: {str(e)}")
            finally:
                self.prefetch_queue.task_done()
    
    def queue_prefetch(self, expert_idx, expert_builder):
        """Queue an expert for prefetching
        
        Args:
            expert_idx: Expert index
            expert_builder: Function that builds the expert model
        """
        if not self.prefetch_enabled:
            return
            
        self.prefetch_queue.put((expert_idx, expert_builder))
        logger.debug(f"Queued expert {expert_idx} for prefetching")
    
    def get_expert(self, expert_idx, expert_builder):
        """
        Get an expert model, loading it into GPU memory if necessary
        
        Args:
            expert_idx: Expert index
            expert_builder: Function that builds the expert model if not cached
        
        Returns:
            Expert model on the target device
        """
        # Update access tracking
        self.access_count[expert_idx] = self.access_count.get(expert_idx, 0) + 1
        self.last_used[expert_idx] = time.time()
        
        # If expert is already in GPU cache, update LRU order and return
        if expert_idx in self.expert_cache:
            # Move to end of LRU list (most recently used)
            self.expert_cache.move_to_end(expert_idx)
            logger.debug(f"Cache hit for expert {expert_idx}")
            return self.expert_cache[expert_idx]
        
        # Expert needs to be loaded
        logger.debug(f"Cache miss for expert {expert_idx}")
        
        # Check if we need to evict an expert to make room
        if len(self.expert_cache) >= self.max_experts:
            self._evict_expert()
        
        # Load expert from CPU cache if available
        if expert_idx in self.cpu_cache:
            logger.debug(f"Loading expert {expert_idx} from CPU cache")
            expert = self.cpu_cache[expert_idx].to(self.device)
            del self.cpu_cache[expert_idx]  # Remove from CPU cache to free memory
        else:
            # Build expert from scratch
            logger.debug(f"Building expert {expert_idx} from scratch")
            expert = expert_builder(expert_idx)
        
        # Add to GPU cache
        self.expert_cache[expert_idx] = expert
        
        # Queue next experts for prefetching
        self._queue_next_experts(expert_idx)
        
        return expert
    
    def _queue_next_experts(self, current_expert_idx):
        """Queue the next most likely experts for prefetching"""
        if not self.prefetch_enabled:
            return
            
        # Use access patterns to predict next experts
        # For now, just prefetch the next few experts in sequence
        num_experts = getattr(self.config, 'num_experts', 8)
        for i in range(1, 4):  # Prefetch next 3 experts
            next_idx = (current_expert_idx + i) % num_experts
            self.queue_prefetch(next_idx, lambda idx: None)  # Placeholder builder
    
    def _evict_expert(self):
        """Evict an expert from GPU cache based on the chosen strategy"""
        if not self.expert_cache:
            return
            
        if self.strategy == "LRU":
            # Evict least recently used expert (first item in OrderedDict)
            expert_idx, expert = next(iter(self.expert_cache.items()))
        elif self.strategy == "FIFO":
            # Evict oldest expert (first item in OrderedDict)
            expert_idx, expert = next(iter(self.expert_cache.items()))
        elif self.strategy == "RANDOM":
            # Evict a random expert
            import random
            expert_idx = random.choice(list(self.expert_cache.keys()))
            expert = self.expert_cache[expert_idx]
        else:
            # Default to LRU
            expert_idx, expert = next(iter(self.expert_cache.items()))
        
        # Move expert to CPU if offloading is enabled
        if self.cpu_offload:
            logger.debug(f"Offloading expert {expert_idx} to CPU")
            self.cpu_cache[expert_idx] = expert.cpu()
        
        # Remove from GPU cache
        logger.debug(f"Evicting expert {expert_idx} from GPU cache")
        del self.expert_cache[expert_idx]
        
    def clear_cache(self):
        """Clear all caches"""
        logger.info("Clearing expert cache")
        self.expert_cache.clear()
        self.cpu_cache.clear()
        torch.cuda.empty_cache()
        
    def shutdown(self):
        """Shutdown prefetch thread and clear caches"""
        logger.info("Shutting down expert cache manager")
        if self.prefetch_thread is not None:
            self.prefetch_queue.put(None)  # Signal thread to exit
            self.prefetch_thread.join(timeout=2.0)
        self.clear_cache() 