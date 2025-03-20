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
                feature_dir = os.path.join(load_path, "features")
                feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".pt")])
                all_features = []
                for feature_file in tqdm(feature_files, desc="Loading feature files"):
                    feature_path = os.path.join(feature_dir, feature_file)
                    individual_features = torch.load(feature_path, map_location=lambda storage, loc: storage.cuda())
                    all_features.append(individual_features)
                features = torch.cat(all_features, dim=0)
            except FileNotFoundError:
                raise FileNotFoundError(f"Features not provided and not found at default path: {load_path}. Please run feature extraction script or provide features directly.")
        assert torch.cuda.is_available(), "CUDA is not available. Clustering cannot run on GPU."
        features = features.cuda()
        assert features.is_cuda, "Features tensor is not on GPU. Clustering will be slow."
        print(f"Features are on GPU: {features.is_cuda}")

        # Stage 1: Fine-grained clustering (paper appendix B, Section 4.1)
        # "cluster these features to 1024 fine-grained centroids"
        # Use multiple GPUs for faster KMeans (if available)
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        gpu_resources = []
        indexes_gpu = []
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for FAISS KMeans.")
            for i in range(num_gpus):
                gpu_resources.append(faiss.GpuResources())
                index_flat = faiss.IndexFlatL2(features.shape[1])
                indexes_gpu.append(faiss.index_cpu_to_gpu(gpu_resources[-1], i, index_flat))
            # Distribute the index across multiple GPUs
            index_gpu = faiss.IndexShards(features.shape[1])
            for sub_index_gpu in indexes_gpu:
                index_gpu.add_shard(sub_index_gpu)
            index_gpu.own_fields = True
        else:
            print("Using single GPU for FAISS KMeans.")
            gpu_resources.append(faiss.GpuResources())
            index_flat = faiss.IndexFlatL2(features.shape[1])
            index_gpu = faiss.index_cpu_to_gpu(gpu_resources[0], 0, index_flat)

        kmeans = faiss.Kmeans(
            features.shape[1],
            self.num_fine, # num_fine_clusters = 1024 as per paper
            niter=100,
            verbose=True,
            spherical=True,  # Paper uses cosine similarity
            nredo=3,  # Paper recommends 3 restarts
            min_points_per_centroid=100,
            max_points_per_centroid=10000,
            index=index_gpu
        )
        assert kmeans.index.is_trained, "KMeans index is not trained (pre-training issue?)"
        assert isinstance(kmeans.index, faiss.GpuIndex), "KMeans index is NOT on GPU!"
        print(f"KMeans index is on GPU: {isinstance(kmeans.index, faiss.GpuIndex)}")

        kmeans.train(features)
        self.fine_centroids = torch.from_numpy(kmeans.centroids).cuda()
        assert self.fine_centroids.is_cuda, "Fine centroids are not on GPU!"

        # Stage 2: Coarse clustering (paper Section 4.1)
        # "then further consolidate to k coarse centroids."
        # "We assign each data point to the nearest of the coarse centroids to produce the final set of partitions."
        # Compute similarities between fine clusters
        similarity_matrix = torch.mm(self.fine_centroids, self.fine_centroids.t()) # Calculate similarity matrix on GPU
        assert similarity_matrix.is_cuda, "Similarity matrix is not on GPU!"

        # Use hierarchical clustering on similarities
        # from scipy.cluster.hierarchy import linkage # CPU-based
        # Z = linkage(similarity_matrix.cpu().numpy(), method='average') # scipy.cluster.hierarchy is CPU-based

        # Cut the dendrogram to get coarse clusters
        # from scipy.cluster.hierarchy import fcluster # CPU-based
        # coarse_labels = fcluster(Z, self.num_coarse, criterion='maxclust') # scipy.cluster.hierarchy is CPU-based

        # Use GPU-accelerated hierarchical clustering from cuML
        similarity_matrix_gpu = similarity_matrix.cuda() # Move similarity matrix to GPU
        assert similarity_matrix_gpu.is_cuda, "Similarity matrix (GPU copy) is not on GPU!"
        agg_clustering = cuml.AgglomerativeClustering(
            n_clusters=self.num_coarse,
            linkage='average', # Use average linkage as in original code
            output_type='pt' # Output as PyTorch tensor for easy integration
        )
        agg_clustering.fit(similarity_matrix_gpu)
        coarse_labels = agg_clustering.labels_ # Keep labels on GPU as PyTorch tensor
        assert coarse_labels.is_cuda, "Coarse labels from cuML are not on GPU!"
        self.coarse_centroids = torch.stack([
            self.fine_centroids[torch.where(coarse_labels == i)[0]].mean(0) # Perform operations on GPU
            for i in range(self.num_coarse) # Iterate through coarse clusters (0 to num_coarse-1)
        ]).cuda()
        assert self.coarse_centroids.is_cuda, "Coarse centroids are not on GPU!"

        # Assign samples to nearest coarse centroid (done in DDMDataset._distribute_samples based on these centroids)
        distances = torch.cdist(features, self.coarse_centroids)
        assert features.is_cuda and self.coarse_centroids.is_cuda, "Features or coarse_centroids are not on GPU for distance calculation!"
        cluster_assignments = torch.argmin(distances, dim=1)
        assert cluster_assignments.is_cuda, "Cluster assignments are not on GPU!"

        # Save individual cluster assignments
        feature_dir = os.path.join(self.default_feature_path, "features")
        cluster_dir = os.path.join(self.default_feature_path, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)
        feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".pt")])  # Ensure consistent order
        num_files = len(feature_files)
        assignments_per_file = len(cluster_assignments) // num_files
        start_index = 0

        for i, feature_file in enumerate(tqdm(feature_files, desc="Saving cluster assignments")):
            end_index = start_index + assignments_per_file
            if i == num_files - 1:  # Handle potential remainder for last file
                end_index = len(cluster_assignments)
            file_assignments = cluster_assignments[start_index:end_index]
            cluster_output_path = os.path.join(cluster_dir, feature_file.replace(".pt", ".cluster.pt"))
            torch.save(file_assignments.cpu(), cluster_output_path)
            start_index = end_index

        return cluster_assignments 