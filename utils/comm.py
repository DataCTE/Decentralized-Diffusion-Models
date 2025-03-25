import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

class QuantizedCommunicator:
    """Implements 8-bit gradient quantization for distributed training"""
    def __init__(self, bits=8, use_symmetric=True):
        self.bits = bits
        self.use_symmetric = use_symmetric
        self.quant_range = 2**(bits-1) - 1  # [-127, 127] for 8-bit
        self._validate_init()
        
    def _validate_init(self):
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized for QuantizedCommunicator")
        if self.bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {self.bits}. Use 4, 8, or 16 bits")
            
    def _get_scale(self, tensor):
        """Calculate scale factor using max absolute value"""
        max_val = tensor.abs().max()
        return max_val / self.quant_range if max_val > 0 else torch.tensor(1.0)

    def quantize(self, tensor):
        """Quantize tensor with dynamic scaling"""
        scale = self._get_scale(tensor)
        scaled_tensor = tensor / (scale + 1e-7)
        quantized = torch.clamp(scaled_tensor.round_(), -self.quant_range, self.quant_range)
        return quantized.to(torch.int8), scale

    def dequantize(self, quantized, scale):
        """Dequantize tensor using stored scale"""
        return quantized.float() * scale

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        """Quantized all-reduce with scaling preservation"""
        try:
            # Quantize the tensor
            quantized, scale = self.quantize(tensor)
            
            # Gather scales from all processes
            scale_tensor = torch.tensor([scale], device=tensor.device)
            dist.all_reduce(scale_tensor, op=dist.ReduceOp.MAX)
            max_scale = scale_tensor.item()
            
            # Normalize and requantize with global scale
            requantized = torch.clamp((tensor / max_scale).round_(), -self.quant_range, self.quant_range)
            
            # Perform quantized all-reduce
            dist.all_reduce(requantized, op=op)
            
            # Dequantize and normalize
            world_size = dist.get_world_size()
            return (requantized.float() * max_scale) / world_size
            
        except Exception as e:
            logger.error(f"Quantized all_reduce failed: {str(e)}")
            # Fallback to regular all-reduce
            dist.all_reduce(tensor, op=op)
            return tensor / dist.get_world_size()

    def broadcast(self, tensor, src=0):
        """Quantized broadcast with scale preservation"""
        if dist.get_rank() == src:
            quantized, scale = self.quantize(tensor)
            dist.broadcast(quantized, src)
            dist.broadcast(torch.tensor([scale], device=tensor.device), src)
            return tensor
        else:
            quantized = torch.empty_like(tensor, dtype=torch.int8)
            scale = torch.tensor([0], device=tensor.device)
            dist.broadcast(quantized, src)
            dist.broadcast(scale, src)
            return self.dequantize(quantized, scale.item()) 