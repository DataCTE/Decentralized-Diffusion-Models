import torch
import faiss
from tqdm import tqdm
import os
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DDMClustering:
    """Implements paper's two-stage clustering from Appendix B and Section 4.1"""
    def __init__(self, num_coarse, num_fine, feature_path=None):
        """
        Initializes the clustering module.

        Args:
            num_coarse (int): Number of coarse clusters (K in the paper).
            num_fine (int): Number of fine-grained clusters for the first stage.
            feature_path (str, optional): Base path for features (might be used for saving centroids). Defaults to None.
        """
        if num_coarse <= 0 or num_fine <= 0:
             raise ValueError("Number of coarse and fine clusters must be positive.")
        if num_fine < num_coarse:
             logger.warning(f"Number of fine clusters ({num_fine}) is less than coarse clusters ({num_coarse}). This is unusual.")

        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.feature_path = feature_path # Store for potential centroid saving
        self.fine_centroids = None
        self.coarse_centroids = None

        # Check for Faiss GPU support
        self.use_gpu = faiss.get_num_gpus() > 0
        if self.use_gpu:
            logger.info("Faiss GPU support detected. Using GPU for KMeans.")
        else:
            logger.info("Faiss GPU support not detected. Using CPU for KMeans.")


    def _train_kmeans(self, features: np.ndarray, k: int, niter: int, spherical: bool = True) -> np.ndarray:
        """ Helper function to train Faiss KMeans on CPU or GPU """
        d = features.shape[1]
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=True, spherical=spherical, gpu=self.use_gpu, nredo=3, min_points_per_centroid=10) # Added reasonable defaults

        # Ensure features are float32 C-contiguous array for Faiss
        features_np = np.ascontiguousarray(features, dtype=np.float32)

        kmeans.train(features_np)
        return kmeans.centroids

    def cluster(self, features: torch.Tensor):
        """
        Performs two-stage clustering on the provided features.

        Args:
            features (torch.Tensor): Tensor of features to cluster, shape (N, D).

        Returns:
            torch.Tensor: Tensor of final coarse cluster assignments, shape (N,). Returns None on failure.
        """
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
             logger.error("Input 'features' must be a 2D PyTorch Tensor.")
             return None
        if features.shape[0] < self.num_fine or features.shape[0] < self.num_coarse:
             logger.error(f"Number of samples ({features.shape[0]}) is less than the number of fine ({self.num_fine}) or coarse ({self.num_coarse}) clusters.")
             return None

        logger.info(f"Starting clustering with features of shape: {features.shape}")
        features_np = features.numpy() # Convert to numpy for Faiss/sklearn

        try:
            # Stage 1: Fine-grained clustering (KMeans)
            logger.info(f"Performing fine-grained KMeans clustering (k={self.num_fine})...")
            self.fine_centroids = self._train_kmeans(features_np, self.num_fine, niter=50) # Reduced iterations for speed potentially
            logger.info(f"Fine-grained KMeans clustering complete. Centroids shape: {self.fine_centroids.shape}")

            if self.fine_centroids.shape[0] < self.num_coarse:
                 logger.error(f"KMeans produced fewer fine centroids ({self.fine_centroids.shape[0]}) than requested coarse clusters ({self.num_coarse}). Cannot proceed with hierarchical clustering.")
                 return None

            # Stage 2: Coarse clustering (Agglomerative on fine centroids)
            logger.info(f"Performing coarse-grained hierarchical clustering (k={self.num_coarse}) on fine centroids...")
            # Calculate similarity (cosine similarity for spherical KMeans) or use Euclidean distance
            # Using Euclidean distance for Agglomerative as similarity matrices can be tricky
            # fine_centroids_tensor = torch.from_numpy(self.fine_centroids).float() # If needed as tensor
            # similarity_matrix = torch.mm(fine_centroids_tensor, fine_centroids_tensor.t()) # Cosine sim if normalized

            # AgglomerativeClustering works on distance matrix (implicitly calculated from features)
            agg_clustering = AgglomerativeClustering(
                n_clusters=self.num_coarse,
                metric='cosine', # Use cosine distance since KMeans was spherical
                linkage='average' # Average linkage often works well
            )
            # Fit on the fine centroids
            agg_clustering.fit(self.fine_centroids) # Pass fine centroids as features
            coarse_labels_for_fine = torch.from_numpy(agg_clustering.labels_).long() # Labels for each fine centroid

            # Calculate coarse centroids (average of fine centroids in each coarse cluster) - Optional, not needed for assignment
            self.coarse_centroids = torch.stack([
                torch.from_numpy(self.fine_centroids[torch.where(coarse_labels_for_fine == i)[0]]).mean(0)
                for i in range(self.num_coarse)
            ]).float()
            logger.info("Coarse-grained hierarchical clustering complete.")
            logger.info(f"Coarse centroids calculated. Shape: {self.coarse_centroids.shape}")


            # Assign original features to the final coarse clusters
            # We need to map each original feature to its nearest *fine* centroid,
            # and then use the coarse label assigned to that fine centroid.
            logger.info("Assigning original features to final coarse clusters...")
            index_fine = faiss.IndexFlatIP(self.fine_centroids.shape[1]) # Use Inner Product (cosine sim) since spherical=True
            res = None
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                index_fine = faiss.index_cpu_to_gpu(res, 0, index_fine)

            index_fine.add(np.ascontiguousarray(self.fine_centroids, dtype=np.float32))
            # Search for the nearest fine centroid for each original feature
            _, fine_centroid_indices = index_fine.search(np.ascontiguousarray(features_np, dtype=np.float32), 1)
            fine_centroid_indices = fine_centroid_indices.squeeze() # Shape (N,)

            # Map fine centroid indices to coarse labels
            final_assignments = coarse_labels_for_fine[fine_centroid_indices] # Shape (N,)
            logger.info("Final cluster assignments generated.")

            # Optional: Save centroids if needed elsewhere
            if self.feature_path:
                 centroids_path = os.path.join(self.feature_path, "centroids")
                 os.makedirs(centroids_path, exist_ok=True)
                 torch.save(self.fine_centroids, os.path.join(centroids_path, "fine_centroids.pt"))
                 torch.save(self.coarse_centroids, os.path.join(centroids_path, "coarse_centroids.pt"))
                 torch.save(coarse_labels_for_fine, os.path.join(centroids_path, "coarse_labels_for_fine.pt")) # Save mapping
                 logger.info(f"Saved centroids to {centroids_path}")


            return final_assignments.long()

        except Exception as e:
            logger.exception(f"An error occurred during clustering: {e}")
            return None
