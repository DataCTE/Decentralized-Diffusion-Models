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
            coarse_clustering = AgglomerativeClustering(
                n_clusters=self.num_coarse,
                metric='cosine', # Changed from 'euclidean'
                linkage='average' # Changed from 'ward' which requires euclidean
            )
            # coarse_clustering = AgglomerativeClustering(n_clusters=self.num_coarse, linkage='ward') # Original
            self.coarse_labels_for_fine = torch.tensor(coarse_clustering.fit_predict(self.fine_centroids), dtype=torch.long) # Shape (num_fine,)

            # Calculate coarse centroids as the mean of their assigned fine centroids
            coarse_means = []
            # fine_centroids_tensor = torch.from_numpy(self.fine_centroids) # Work with tensors - Moved conversion down
            # Convert to tensor here, after it's used as numpy by AgglomerativeClustering
            fine_centroids_tensor = torch.from_numpy(self.fine_centroids) 
            for i in range(self.num_coarse):
                fine_indices = torch.where(self.coarse_labels_for_fine == i)[0]
                if len(fine_indices) > 0:
                     # Use the tensor version of fine_centroids for mean calculation
                     mean_vec = torch.mean(fine_centroids_tensor[fine_indices], dim=0)
                     coarse_means.append(mean_vec)
                else:
                     logger.warning(f"Coarse cluster {i} is empty. Assigning zero vector as centroid.")
                     # Create a zero vector with the same shape and type as other centroids
                     zero_vec = torch.zeros_like(fine_centroids_tensor[0]) 
                     coarse_means.append(zero_vec)
            
            if len(coarse_means) != self.num_coarse:
                logger.error(f"Expected {self.num_coarse} coarse centroids, but generated {len(coarse_means)}. This indicates a logic error.")
                return None 

            self.coarse_centroids = torch.stack(coarse_means) # Shape (num_coarse, D)
            logger.info(f"Coarse clustering complete. Coarse centroids shape: {self.coarse_centroids.shape}")

            # --- MODIFICATION START: Assign original features to NEAREST COARSE centroid ---
            logger.info("Assigning original features to final coarse clusters (nearest coarse centroid)...")
            
            # Ensure coarse centroids are numpy float32 for Faiss
            coarse_centroids_np = np.ascontiguousarray(self.coarse_centroids.numpy(), dtype=np.float32)
            features_np_cont = np.ascontiguousarray(features_np, dtype=np.float32)
            
            # Create Faiss index for COARSE centroids (Use Inner Product for cosine similarity)
            index_coarse = faiss.IndexFlatIP(coarse_centroids_np.shape[1]) 
            res_coarse = None
            if self.use_gpu:
                res_coarse = faiss.StandardGpuResources()
                index_coarse = faiss.index_cpu_to_gpu(res_coarse, 0, index_coarse)

            index_coarse.add(coarse_centroids_np)
            
            # Search the COARSE index using the ORIGINAL features
            _, final_assignments_indices = index_coarse.search(features_np_cont, 1)
            final_assignments = torch.from_numpy(final_assignments_indices.squeeze()).long() # Shape (N,)
            # --- MODIFICATION END ---

            logger.info("Final cluster assignments generated.")

            # Optional: Save centroids if needed elsewhere
            if self.feature_path:
                 centroids_path = os.path.join(self.feature_path, "centroids")
                 os.makedirs(centroids_path, exist_ok=True)
                 # Save fine_centroids as numpy array directly from Faiss KMeans output
                 np.save(os.path.join(centroids_path, "fine_centroids.npy"), self.fine_centroids)
                 # Save coarse centroids and mapping as tensors
                 torch.save(self.coarse_centroids, os.path.join(centroids_path, "coarse_centroids.pt"))
                 torch.save(self.coarse_labels_for_fine, os.path.join(centroids_path, "coarse_labels_for_fine.pt"))
                 logger.info(f"Saved centroids to {centroids_path}")


            return final_assignments # Return the direct coarse assignments

        except Exception as e:
            logger.exception(f"An error occurred during clustering: {e}")
            return None
