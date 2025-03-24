import torch
import faiss
from tqdm import tqdm
import os
from sklearn.cluster import AgglomerativeClustering

class DDMClustering:
    """Implements paper's two-stage clustering from Appendix B and Section 4.1"""
    def __init__(self, num_coarse, num_fine):
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fine_centroids = None
        self.coarse_centroids = None
        self.default_feature_path = "/home/alex/workspace/Decentralized-Diffusion-Models/features"

    def cluster(self, features):
        """Paper's two-stage clustering (Appendix B)"""
        # Stage 1: Fine-grained k-means
        kmeans = faiss.Kmeans(
            features.shape[1], 
            self.num_fine,
            niter=100,
            verbose=False,
            spherical=True,
            gpu=True
        )
        kmeans.train(features)
        
        # Stage 2: Hierarchical merging
        agg = AgglomerativeClustering(
            n_clusters=self.num_coarse,
            linkage='average',
            metric='cosine'
        )
        agg.fit(kmeans.centroids)
        
        # Assign original points to coarse clusters
        _, fine_labels = kmeans.index.search(features, 1)
        coarse_labels = agg.labels_[fine_labels.flatten()]
        
        return coarse_labels

    def cluster(self, features_list=None, feature_path=None):
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
        else:
            features = torch.cat(features_list, dim=0).float()

        print(f"Starting clustering with features of shape: {features.shape}")

        # Stage 1: Fine-grained clustering
        print("Performing fine-grained KMeans clustering on CPU...")
        kmeans = faiss.Kmeans(
            features.shape[1],
            self.num_fine,
            niter=100,
            verbose=True,
            spherical=True,
            nredo=3,
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
        )
        kmeans.train(features.numpy())
        self.fine_centroids = torch.from_numpy(kmeans.centroids).float()
        print("Fine-grained KMeans clustering complete.")

        # Stage 2: Coarse clustering
        print("Performing coarse-grained hierarchical clustering on CPU...")
        similarity_matrix = torch.mm(self.fine_centroids, self.fine_centroids.t())

        agg_clustering = AgglomerativeClustering(
            n_clusters=self.num_coarse,
            linkage='average'
        )
        agg_clustering.fit(similarity_matrix.numpy())
        coarse_labels = torch.from_numpy(agg_clustering.labels_).long()
        self.coarse_centroids = torch.stack([
            self.fine_centroids[torch.where(coarse_labels == i)[0]].mean(0)
            for i in range(self.num_coarse)
        ])
        print("Coarse-grained hierarchical clustering complete.")

        distances = torch.cdist(features, self.coarse_centroids)
        cluster_assignments = torch.argmin(distances, dim=1).long()

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
