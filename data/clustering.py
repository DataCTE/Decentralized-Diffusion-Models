import torch
import faiss
from tqdm import tqdm
import os
import cuml

class DDMClustering:
    """Implements paper's two-stage clustering from Appendix B and Section 4.1"""
    def __init__(self, num_coarse_clusters=8, num_fine_clusters=1024, default_feature_path="/home/alex/workspace/Decentralized-Diffusion-Models/features"):
        self.num_coarse = num_coarse_clusters
        self.num_fine = num_fine_clusters
        self.fine_centroids = None
        self.coarse_centroids = None
        self.default_feature_path = default_feature_path

    def cluster(self, features_list=None, feature_path=None):
        """
        Args:
            features_list: List of Tensors, each of shape [N_i, D] containing DINOv2 features on different GPUs.
                           If None, it will attempt to load features from feature_path or default_feature_path.
            feature_path: Path to load features from if features_list is None. Defaults to default_feature_path if None.
        Returns:
            cluster_assignments: Tensor of shape [N] with coarse cluster IDs
        """
        if features_list is None:
            load_path = feature_path if feature_path else self.default_feature_path
            print(f"Loading features from default path: {load_path}")
            try:
                feature_dir = os.path.join(load_path, "features")
                feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".pt")])
                all_features = []
                for feature_file in tqdm(feature_files, desc="Loading feature files"):
                    feature_path = os.path.join(feature_dir, feature_file)
                    individual_features = torch.load(feature_path, map_location=lambda storage, loc: storage.cpu())
                    all_features.append(individual_features)
                features = torch.cat(all_features, dim=0).float()
                features_list = [features]
            except FileNotFoundError:
                raise FileNotFoundError(f"Features not provided and not found at default path: {load_path}. Please run feature extraction script or provide features directly.")

        print(f"Starting clustering with features of shape: {features.shape}")

        # Stage 1: Fine-grained clustering (paper appendix B, Section 4.1)
        # "cluster these features to 1024 fine-grained centroids"
        # Use FAISS KMeans on CPU for simplicity and debugging
        print("Performing fine-grained KMeans clustering on CPU...")
        kmeans = faiss.Kmeans(
            features.shape[1],
            self.num_fine, # num_fine_clusters = 1024 as per paper
            niter=100,
            verbose=True,
            spherical=True,  # Paper uses cosine similarity
            nredo=3,  # Paper recommends 3 restarts
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
            gpu_index=False
        )
        kmeans.train(features.numpy())
        self.fine_centroids = torch.from_numpy(kmeans.centroids).float()

        print("Fine-grained KMeans clustering complete.")

        # Stage 2: Coarse clustering (paper Section 4.1)
        # "then further consolidate to k coarse centroids."
        # "We assign each data point to the nearest of the coarse centroids to produce the final set of partitions."
        print("Performing coarse-grained hierarchical clustering on CPU...")

        # Compute similarities between fine clusters on CPU
        similarity_matrix = torch.mm(self.fine_centroids, self.fine_centroids.t())

        # Use CPU-based hierarchical clustering from cuML (single CPU for now)
        agg_clustering = cuml.AgglomerativeClustering(
            n_clusters=self.num_coarse,
            linkage='average', # Use average linkage as in original code
            output_type='pt' # Output as PyTorch tensor for easy integration
        )
        agg_clustering.fit(similarity_matrix.numpy())
        coarse_labels = torch.from_numpy(agg_clustering.labels_).long()
        self.coarse_centroids = torch.stack([
            self.fine_centroids[torch.where(coarse_labels == i)[0]].mean(0)
            for i in range(self.num_coarse)
        ])

        print("Coarse-grained hierarchical clustering complete.")

        # Assign samples to nearest coarse centroid (done in DDMDataset._distribute_samples based on these centroids)
        distances = torch.cdist(features, self.coarse_centroids)
        cluster_assignments = torch.argmin(distances, dim=1).long()

        # Save individual cluster assignments
        feature_dir = os.path.join(self.default_feature_path, "features")
        cluster_dir = os.path.join(self.default_feature_path, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".pt")])
        num_files = len(feature_files)
        assignments_per_file = len(cluster_assignments) // num_files
        start_index = 0

        for i, feature_file in enumerate(tqdm(feature_files, desc="Saving cluster assignments")):
            end_index = start_index + assignments_per_file
            if i == num_files - 1:
                end_index = len(cluster_assignments)
            file_assignments = cluster_assignments[start_index:end_index]
            cluster_output_path = os.path.join(cluster_dir, feature_file.replace(".pt", ".cluster.pt"))
            torch.save(file_assignments, cluster_output_path)
            start_index = end_index

        return cluster_assignments 