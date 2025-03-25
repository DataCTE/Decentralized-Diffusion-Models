class QuantizedCommunicator:
    def __init__(self, bits=8):
        self.bits = bits
        self.quantizer = torch.quantization.QuantStub()
        self.dequantizer = torch.quantization.DeQuantStub()

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        # Quantize before communication
        q_tensor = self.quantizer(tensor)
        dist.all_reduce(q_tensor, op=op)
        return self.dequantizer(q_tensor) 