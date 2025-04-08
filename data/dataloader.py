import torch
import torch.nn.functional as F
import gc # Import garbage collector

class AdaptiveBucketSampler:
    def __init__(self, buckets):
        # Paper's adaptive probability weights
        self.sample_weights = torch.ones(len(buckets))
        self.update_frequency = 1000  # Steps between updates
        self.temperature = 1.0  # Initial temperature
        
    def update_weights(self, model_performance):
        # Paper's adaptive probability weights with softmax temperature
        try: # Add try/finally for cleanup
            self.sample_weights = F.softmax(model_performance / self.temperature, dim=-1)
            self.temperature = max(0.1, self.temperature * 0.95)  # Anneal temperature 
        finally:
            # --- Explicitly delete input tensor reference --- START EDIT ---
            # Although it's an argument, explicitly deleting might help 
            # if references are held elsewhere unexpectedly.
            del model_performance 
            # gc.collect() is likely overkill here unless model_performance is huge
            # and this is called very frequently causing fragmentation.
            # gc.collect() 
            # --- Explicitly delete input tensor reference --- END EDIT --- 