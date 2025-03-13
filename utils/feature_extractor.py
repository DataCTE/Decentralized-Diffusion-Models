"""Feature extraction for DDM clustering (Paper Section 3.2)"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class DINOv2FeatureExtractor(nn.Module):
    """Implements paper's feature extraction using DINOv2"""
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_dinov2()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def _init_dinov2(self):
        """Load DINOv2 model with paper-recommended settings"""
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.no_grad()
    def forward(self, x):
        """Paper's feature extraction forward pass"""
        # Normalize input
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        
        # Extract features
        features = self.model.forward_features(x)['x_norm_patchtokens']
        
        # Average pooling across spatial dimensions
        return features.mean(dim=1)

    def extract_features(self, dataloader):
        """Paper's Algorithm 1: Feature extraction from dataset"""
        self.model.eval()
        features = []
        
        for batch in tqdm(dataloader, desc="Extracting DINOv2 features"):
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch
                
            images = images.to(self.device)
            with torch.cuda.amp.autocast():
                batch_features = self(images)
            features.append(batch_features.cpu())
            
        return torch.cat(features).numpy()
