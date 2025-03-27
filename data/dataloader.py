import torch
import torch.nn.functional as F

class AdaptiveBucketSampler:
    def __init__(self, buckets):
        # Paper's adaptive probability weights
        self.sample_weights = torch.ones(len(buckets))
        self.update_frequency = 1000  # Steps between updates
        self.temperature = 1.0  # Initial temperature
        
    def update_weights(self, model_performance):
        # Paper's adaptive probability weights with softmax temperature
        self.sample_weights = F.softmax(model_performance / self.temperature, dim=-1)
        self.temperature = max(0.1, self.temperature * 0.95)  # Anneal temperature 