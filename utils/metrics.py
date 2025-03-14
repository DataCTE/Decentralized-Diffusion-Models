"""Metrics utilities for Decentralized Diffusion Models."""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MetricCalculator:
    """Paper-specified metrics from Section 4.3"""
    
    @staticmethod
    def fid(real_features, gen_features):
        """
        Frechet Inception Distance as per paper Eq. 9
        Args:
            real_features: [N, D] tensor from real data
            gen_features: [N, D] tensor from generated samples
        """
        # Convert to numpy if tensors
        if isinstance(real_features, torch.Tensor):
            real_features = real_features.cpu().numpy()
        if isinstance(gen_features, torch.Tensor):
            gen_features = gen_features.cpu().numpy()
            
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_gen
        # Product might not be invertible, add small identity matrix to make it more stable
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        # Numerical issues might make complex part != 0
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
        return float(fid_score)

    @staticmethod
    def clip_score(images, text_embeddings, clip_model):
        """
        CLIP Score from paper Appendix B.3
        Args:
            images: [N, C, H, W] tensor of generated images
            text_embeddings: [N, D] CLIP text features
            clip_model: Pre-trained CLIP model
        """
        # Get image features
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (image_features * text_features).sum(dim=1).mean()
        return similarity.item()
        
    @staticmethod
    def inception_score(features, splits=10):
        """
        Calculate Inception Score
        Args:
            features: [N, 1000] tensor of softmax probabilities
            splits: Number of splits for bootstrapping
        """
        # Convert to numpy if tensor
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            
        # Split into chunks for bootstrapping
        chunk_size = features.shape[0] // splits
        scores = []
        
        for i in range(splits):
            chunk = features[i * chunk_size:(i + 1) * chunk_size]
            
            # Calculate marginal and conditional distributions
            marginal = np.mean(chunk, axis=0)
            kl = chunk * (np.log(chunk + 1e-10) - np.log(marginal + 1e-10))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)
        
    @staticmethod
    def calculate_all_metrics(generated_images, real_features, clip_model=None, text_prompts=None):
        """
        Calculate all metrics for generated images
        Args:
            generated_images: Tensor of generated images
            real_features: Features from real dataset
            clip_model: CLIP model for text-image alignment (optional)
            text_prompts: Text prompts used for generation (optional)
        """
        metrics = {}
        
        # Get inception model
        inception = inception_v3(pretrained=True, transform_input=False).eval()
        
        # Calculate inception features for generated images
        with torch.no_grad():
            gen_features = inception(generated_images)
            
        # Calculate FID
        metrics['fid'] = MetricCalculator.fid(real_features, gen_features)
        
        # Calculate Inception Score
        is_mean, is_std = MetricCalculator.inception_score(F.softmax(gen_features, dim=1))
        metrics['inception_score'] = is_mean
        metrics['inception_score_std'] = is_std
        
        # Calculate CLIP Score if available
        if clip_model is not None and text_prompts is not None:
            text_embeddings = clip_model.encode_text(text_prompts)
            metrics['clip_score'] = MetricCalculator.clip_score(
                generated_images, text_embeddings, clip_model
            )
            
        return metrics
        
    @staticmethod
    def extract_inception_features(dataloader, max_samples=None, device='cuda'):
        """
        Extract features from a dataset using Inception v3
        
        Args:
            dataloader: DataLoader for dataset
            max_samples: Maximum number of samples to process (optional)
            device: Device to run extraction on
            
        Returns:
            [N, 2048] numpy array of features
        """
        # Load inception model
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()  # Remove classification layer
        inception = inception.to(device).eval()
        
        # Extract features
        features = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if isinstance(batch, dict):
                    images = batch.get('image', batch.get('images', None))
                else:
                    images = batch
                    
                # Check if we have a valid tensor
                if not isinstance(images, torch.Tensor):
                    logger.warning(f"Unexpected batch format: {type(batch)}")
                    continue
                    
                # Move to device
                images = images.to(device)
                
                # Extract features
                batch_features = inception(images)
                features.append(batch_features.cpu())
                
                # Update sample count
                sample_count += images.size(0)
                
                # Check if we've reached the maximum
                if max_samples is not None and sample_count >= max_samples:
                    break
                    
        # Concatenate features
        features = torch.cat(features, dim=0)
        
        # Limit to max_samples if specified
        if max_samples is not None:
            features = features[:max_samples]
            
        return features.numpy()
        
    @staticmethod
    def psnr(img1, img2):
        """
        Calculate Peak Signal-to-Noise Ratio between two images
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            
        Returns:
            PSNR value
        """
        mse = F.mse_loss(img1, img2, reduction='mean')
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
        
    @staticmethod
    def ssim(img1, img2):
        """
        Calculate Structural Similarity Index between two images
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            
        Returns:
            SSIM value
        """
        import pytorch_msssim
        ssim_module = pytorch_msssim.SSIM(data_range=1.0, size_average=True, channel=3)
        return ssim_module(img1, img2).item()
        
    @staticmethod
    def lpips(img1, img2, lpips_model=None):
        """
        Calculate Learned Perceptual Image Patch Similarity
        
        Args:
            img1: First image tensor [B, C, H, W]
            img2: Second image tensor [B, C, H, W]
            lpips_model: LPIPS model (will be loaded if None)
            
        Returns:
            LPIPS distance
        """
        try:
            import lpips
            if lpips_model is None:
                lpips_model = lpips.LPIPS(net='alex').to(img1.device)
            
            with torch.no_grad():
                distance = lpips_model(img1, img2)
            return distance.mean().item()
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")
            return None