import torch
import faiss
from tqdm import tqdm

class DDMClustering:
    """Implements paper's two-stage clustering from Appendix B"""
    def __init__(self, num_coarse_clusters=8, num_fine_clusters=1024):
        self.num_coarse = num_coarse_clusters
        self.num_fine = num_fine_clusters
        self.fine_centroids = None
        self.coarse_centroids = None

    def cluster(self, features):
        """
        Args:
            features: Tensor of shape [N, D] containing DINOv2 features
        Returns:
            cluster_assignments: Tensor of shape [N] with coarse cluster IDs
        """
        # Stage 1: Fine-grained clustering (paper appendix B)
        kmeans = faiss.Kmeans(
            features.shape[1],
            self.num_fine,
            niter=100,
            verbose=True,
            gpu=True,
            spherical=True,  # Paper uses cosine similarity
            nredo=3,  # Paper recommends 3 restarts
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
        )
        kmeans.train(features.cpu().numpy())
        self.fine_centroids = torch.from_numpy(kmeans.centroids).cuda()

        # Stage 2: Coarse clustering
        # Compute similarities between fine clusters
        similarity_matrix = torch.mm(
            self.fine_centroids, 
            self.fine_centroids.t()
        )
        
        # Use hierarchical clustering on similarities
        from fastcluster import linkage
        Z = linkage(similarity_matrix.cpu().numpy(), method='average')
        
        # Cut the dendrogram to get coarse clusters
        from scipy.cluster.hierarchy import fcluster
        coarse_labels = fcluster(Z, self.num_coarse, criterion='maxclust')
        self.coarse_centroids = torch.stack([
            self.fine_centroids[torch.where(torch.from_numpy(coarse_labels) == i)[0]].mean(0)
            for i in range(1, self.num_coarse+1)
        ])

        # Assign samples to nearest coarse centroid
        distances = torch.cdist(features, self.coarse_centroids)
        return torch.argmin(distances, dim=1) 