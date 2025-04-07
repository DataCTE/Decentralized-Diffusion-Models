import logging
import sys
import torch.distributed as dist
import os

# Basic distributed check functions (could be moved to a central place if used more widely)
def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True # Not distributed or not initialized, assume main process
    return dist.get_rank() == 0

def get_rank():
    """Gets the rank of the current process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0 # Default to rank 0 if not distributed/initialized
    return dist.get_rank()

def setup_distributed_logger(name="distributed_logger", level=logging.INFO, rank=None):
    """
    Sets up a logger with rank-specific formatting and level.

    Args:
        name (str): The name for the logger.
        level (int): The base logging level (e.g., logging.INFO).
                    Non-main processes will have their level increased to WARNING.
        rank (int, optional): The rank of the current process. If None, attempts to get it
                              using `get_rank()`.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if rank is None:
        rank = get_rank() # Get rank if not provided

    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if already configured
    if logger.hasHandlers():
        return logger

    # Set level: INFO for main process, WARNING for others by default
    log_level = level if rank == 0 else max(level, logging.WARNING)
    logger.setLevel(log_level)

    # Create handler (log to stderr)
    handler = logging.StreamHandler(stream=sys.stderr)

    # Create formatter including rank
    formatter = logging.Formatter(
        f'%(asctime)s - Rank {rank} - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)

    # Optional: Prevent propagation to root logger if desired
    # logger.propagate = False

    return logger

