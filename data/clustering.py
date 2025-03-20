import torch
import faiss
from tqdm import tqdm
import os

class DDMClustering:
    """Implements paper's two-stage clustering from Appendix B and Section 4.1"""
    def __init__(self, num_coarse_clusters=8, num_fine_clusters=1024, default_feature_path="/home/alex/workspace/Decentralized-Diffusion-Models/features"):
        self.num_coarse = num_coarse_clusters
        self.num_fine = num_fine_clusters
        self.fine_centroids = None
        self.coarse_centroids = None
        self.default_feature_path = default_feature_path

    def cluster(self, features=None, feature_path=None):
        """
        Args:
            features: Tensor of shape [N, D] containing DINOv2 features (pre-computed, see paper Section 4.1).
                      If None, it will attempt to load features from feature_path or default_feature_path.
            feature_path: Path to load features from if features is None. Defaults to default_feature_path if None.
        Returns:
            cluster_assignments: Tensor of shape [N] with coarse cluster IDs
        """
        if features is None:
            load_path = feature_path if feature_path else self.default_feature_path
            print(f"Loading features from default path: {load_path}")
            try:
                features = torch.load(os.path.join(load_path, "train_features.pt"))
            except FileNotFoundError:
                raise FileNotFoundError(f"Features not provided and not found at default path: {load_path}. Please run feature extraction script or provide features directly.")
        # Stage 1: Fine-grained clustering (paper appendix B, Section 4.1)
        # "cluster these features to 1024 fine-grained centroids"
        kmeans = faiss.Kmeans(
            features.shape[1],
            self.num_fine, # num_fine_clusters = 1024 as per paper
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

        # Stage 2: Coarse clustering (paper Section 4.1)
        # "then further consolidate to k coarse centroids."
        # "We assign each data point to the nearest of the coarse centroids to produce the final set of partitions."
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

        # Assign samples to nearest coarse centroid (done in DDMDataset._distribute_samples based on these centroids)
        distances = torch.cdist(features, self.coarse_centroids)
        return torch.argmin(distances, dim=1) 