"""Expert Cache Manager for memory-efficient handling of multiple expert models."""

import torch
import logging
import time
import threading
from collections import OrderedDict
from queue import Queue, Full

logger = logging.getLogger(__name__)

class ExpertCacheManager:
    """
    Manages loading and unloading of expert models to efficiently use GPU memory.
    Implements various caching strategies (LRU, FIFO, RANDOM) with prefetching support.
    """
    
    def __init__(self, config, device="cuda", max_experts=None, swap_strategy=None, 
                 cpu_offload=None, prefetch=None):
        """
        Initialize expert cache manager
        
        Args:
            config: Configuration object with expert caching settings
            device: Device to load experts on
            max_experts: Maximum number of experts to keep in memory
            swap_strategy: Strategy for swapping experts ("LRU", "FIFO", "RANDOM")
            cpu_offload: Whether to offload experts to CPU
            prefetch: Whether to prefetch experts
        """
        self.config = config
        self.device = device
        self.max_experts = max_experts or getattr(config, 'max_experts_in_memory', 3)
        self.strategy = swap_strategy or getattr(config, 'expert_swap_strategy', "LRU")
        self.cpu_offload = cpu_offload if cpu_offload is not None else getattr(config, 'expert_offload_to_cpu', True)
        self.prefetch_enabled = prefetch if prefetch is not None else getattr(config, 'expert_prefetch_next', True)
        
        # Cache and usage tracking
        self.expert_cache = OrderedDict()  # Maps expert_idx -> expert model
        self.cpu_cache = {}  # Maps expert_idx -> CPU copy of expert model
        self.access_count = {}  # Maps expert_idx -> number of accesses
        self.pending_prefetch = set()  # Track which experts are being prefetched
        
        # For LRU strategy
        self.last_used = {}  # Maps expert_idx -> timestamp of last use
        
        # Memory tracking
        self.expert_memory_usage = {}  # Maps expert_idx -> estimated memory usage
        
        # Lock for thread-safe operations
        self.cache_lock = threading.RLock()
        
        # Prefetching with bounded queue to limit memory pressure
        self.prefetch_queue_size = getattr(config, 'prefetch_queue_size', 2)
        self.prefetch_queue = Queue(maxsize=self.prefetch_queue_size)
        self.prefetch_thread = None
        
        # Start prefetch thread if enabled
        if self.prefetch_enabled:
            self._start_prefetch_thread()
            
        logger.info(f"Initialized ExpertCacheManager with strategy={self.strategy}, max_experts={self.max_experts}")
    
    def _start_prefetch_thread(self):
        """Start background thread for prefetching experts"""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            return  # Thread already running
            
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.info("Started expert prefetching thread")
        
    def _prefetch_worker(self):
        """Worker thread that prefetches experts to CPU or GPU"""
        while True:
            try:
                # Get expert to prefetch
                expert_info = self.prefetch_queue.get()
                if expert_info is None:  # Exit signal
                    break
                    
                expert_idx, expert_builder = expert_info
                
                try:
                    with self.cache_lock:
                        # Skip if already cached
                        if expert_idx in self.cpu_cache or expert_idx in self.expert_cache:
                            self.pending_prefetch.discard(expert_idx)
                            self.prefetch_queue.task_done()
                            continue
                        
                        # Build expert and move to CPU
                        expert = expert_builder(expert_idx)
                        
                        # Estimate memory usage if not already known
                        if expert_idx not in self.expert_memory_usage:
                            self.expert_memory_usage[expert_idx] = sum(
                                p.numel() * p.element_size() 
                                for p in expert.parameters()
                            )
                        
                        if self.cpu_offload:
                            # Move to CPU
                            expert = expert.cpu()
                            self.cpu_cache[expert_idx] = expert
                            logger.debug(f"Prefetched expert {expert_idx} to CPU cache")
                        else:
                            # Check if we have room in GPU cache
                            if len(self.expert_cache) < self.max_experts:
                                self.expert_cache[expert_idx] = expert
                                logger.debug(f"Prefetched expert {expert_idx} directly to GPU cache")
                            else:
                                # No room in GPU, store in CPU cache
                                expert = expert.cpu()
                                self.cpu_cache[expert_idx] = expert
                                logger.debug(f"Prefetched expert {expert_idx} to CPU cache (GPU full)")
                        
                        # Remove from pending set
                        self.pending_prefetch.discard(expert_idx)
                
                except Exception as e:
                    logger.error(f"Error prefetching expert {expert_idx}: {str(e)}")
                    with self.cache_lock:
                        self.pending_prefetch.discard(expert_idx)
                
                finally:
                    self.prefetch_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in prefetch worker: {str(e)}")
    
    def get_expert(self, expert_idx, expert_builder):
        """
        Get an expert model, loading it if necessary
        
        Args:
            expert_idx: Expert index
            expert_builder: Function to build the expert if not cached
            
        Returns:
            Expert model on the target device
        """
        with self.cache_lock:
            # Track access statistics
            self.access_count[expert_idx] = self.access_count.get(expert_idx, 0) + 1
            self.last_used[expert_idx] = time.time()
            
            # Check if the expert is already in GPU cache
            if expert_idx in self.expert_cache:
                # Update LRU order by removing and re-adding
                expert = self.expert_cache.pop(expert_idx)
                self.expert_cache[expert_idx] = expert
                logger.debug(f"Using cached expert {expert_idx} from GPU")
                return expert
        
            # If we need to load, first check if we need to make room
            if len(self.expert_cache) >= self.max_experts:
                self._evict_expert()
            
            # Load expert from CPU cache if available
            if expert_idx in self.cpu_cache:
                logger.debug(f"Loading expert {expert_idx} from CPU cache")
                expert = self.cpu_cache[expert_idx].to(self.device)
                del self.cpu_cache[expert_idx]  # Remove from CPU cache to free memory
                torch.cuda.empty_cache()  # Ensure CUDA memory is freed
            else:
                # Build expert from scratch
                logger.debug(f"Building expert {expert_idx} from scratch")
                expert = expert_builder(expert_idx)
                
                # Estimate memory usage
                self.expert_memory_usage[expert_idx] = sum(
                    p.numel() * p.element_size() 
                    for p in expert.parameters()
                )
            
            # Add to GPU cache
            self.expert_cache[expert_idx] = expert
            
            # Queue next experts for prefetching
            self._queue_next_experts(expert_idx)
            
            return expert
    
    def queue_prefetch(self, expert_idx, expert_builder):
        """Queue an expert for prefetching"""
        if not self.prefetch_enabled:
            return False
            
        # Check if already in cache or pending
        with self.cache_lock:
            if (expert_idx in self.expert_cache or 
                expert_idx in self.cpu_cache or 
                expert_idx in self.pending_prefetch):
                return False
                
            # Mark as pending
            self.pending_prefetch.add(expert_idx)
        
        # Add to prefetch queue
        try:
            self.prefetch_queue.put((expert_idx, expert_builder), block=False)
            return True
        except Full:
            # Queue is full, remove from pending
            with self.cache_lock:
                self.pending_prefetch.discard(expert_idx)
            return False
    
    def _queue_next_experts(self, current_expert_idx):
        """Queue the next most likely experts for prefetching"""
        if not self.prefetch_enabled:
            return
            
        # Use access patterns to predict next experts
        # For now, just prefetch the next few experts in sequence
        num_experts = getattr(self.config, 'num_experts', 8)
        for i in range(1, min(4, num_experts)):  # Prefetch next 3 experts or fewer
            next_idx = (current_expert_idx + i) % num_experts
            self.queue_prefetch(next_idx, lambda idx: None)  # Placeholder builder
    
    def _evict_expert(self):
        """Evict an expert from GPU cache based on the chosen strategy"""
        if not self.expert_cache:
            return
            
        if self.strategy == "LRU":
            # Evict least recently used expert
            expert_idx, expert = next(iter(self.expert_cache.items()))
            # Find expert with minimum last_used timestamp
            if self.last_used:
                expert_idx = min(
                    self.expert_cache.keys(),
                    key=lambda idx: self.last_used.get(idx, 0)
                )
                expert = self.expert_cache[expert_idx]
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
        torch.cuda.empty_cache()  # Ensure CUDA memory is freed
        
    def get_memory_estimate(self, expert_idx):
        """Get estimated memory usage for an expert"""
        return self.expert_memory_usage.get(expert_idx, 0)
        
    def clear_cache(self):
        """Clear all caches"""
        logger.info("Clearing expert cache")
        with self.cache_lock:
            self.expert_cache.clear()
            self.cpu_cache.clear()
        torch.cuda.empty_cache()
        
    def shutdown(self):
        """Shutdown prefetch thread and clear caches"""
        logger.info("Shutting down expert cache manager")
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_queue.put(None)  # Signal thread to exit
            self.prefetch_thread.join(timeout=2.0)
        self.clear_cache() 