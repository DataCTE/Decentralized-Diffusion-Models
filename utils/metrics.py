import torch

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
        mu_real, sigma_real = real_features.mean(0), torch.cov(real_features.T)
        mu_gen, sigma_gen = gen_features.mean(0), torch.cov(gen_features.T)
        
        # FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^½)
        diff = mu_real - mu_gen
        cov_mean = torch.linalg.matrix_power(sigma_real @ sigma_gen, 0.5)
        return diff.dot(diff) + torch.trace(sigma_real + sigma_gen - 2*cov_mean)

    @staticmethod
    def clip_score(images, text_embeddings, clip_model):
        """
        CLIP Score from paper Appendix B.3
        Args:
            images: [N, C, H, W] tensor of generated images
            text_embeddings: [N, D] CLIP text features
            clip_model: Pre-trained CLIP model
        """
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=1).mean()