import numpy
from tqdm.auto import tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


class MiniVec2Vec:
    def __init__(self):
        """
        Constructor for the class.
        """
        self.W = None

    def fit(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int = 20,
        n_runs: int = 30,
        top_k: int = 50,
        subsample: float = 0.33,
        random_seed: int = 123,
        verbose: bool = True,
    ):
        """
        Create the optimal matrix `W` of linear transforms for
        mapping embeddings from embedding space A to embedding
        space B.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_clusters (int, Optional):
                Number of clusters for k-means for finding centroids
                Default is 20

            n_runs (int, Optional):
                Number of iterations to run to get a varied set of
                centers that represent the embedding spaces.
                Default is 30

            top_k (int, Optional):
                In the relative representation space for embedding in A and B,
                find the top k nearest neighbors from relative space from B for each A.
                (A and B share the same representation space)

            subsample (float or None, Optional):
                For each iteration use a percentage subset of the
                data to in A and B, to get the centers.
                Value must be Between 0 and 1.
                If 1 use all the data.
                Default is 0.33

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Default is 123

            verbose (bool, Optional):
                If False it will hide all the progress bars
                Default is True

        Returns: self
            Fitted MiniVec2Vec object
        """
        # --- Set Up ---

        # Check subsample arg
        if isinstance(subsample, float) and not 0 < subsample <= 1:
            raise ValueError(f"`subsample` value, {subsample}, must be between 0 and 1")

        # Check input embeddings dims
        assert A.ndim == 2
        assert B.ndim == 2

        # Preprocess embeddings s.t. they center around the mean and normalise
        A = normalize(A - A.mean(axis=0))
        B = normalize(B - B.mean(axis=0))

        # Set random seed
        rng = numpy.random.default_rng(random_seed)

        # --- Match Anchors ---

        # get centroids
        A_centers = []
        B_centers = []
        for _ in tqdm(range(n_runs), desc="Matching Anchors ...", disable=not verbose):
            # Use subsample and K-Means
            A_clusters, B_clusters = self._align_centroids(
                A, B, n_clusters, n_runs, subsample, rng
            )
            A_centers.append(normalize(A_clusters))
            B_centers.append(normalize(B_clusters))

        # get top k similar from cosine sim between centers and input data
        nn = NearestNeighbors(
            n_neighbors=top_k,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(B @ numpy.vstack(B_centers).T)
        _, top_similar = nn.kneighbors(A @ numpy.vstack(A_centers).T)

        # TODO: Understand
        Y_matched = B[top_similar].swapaxes(-1, -2) @ (numpy.ones(top_k) / top_k)

        # --- Train Mappings ---
        self.W, _ = orthogonal_procrustes(A, Y_matched)

        # --- Refinement 1: Iterative Closest Point Average ---
        # Fit KNN on B but also overwrite nn var to save memory
        nn = NearestNeighbors(
            n_neighbors=top_k,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(B)
        for _ in tqdm(range(n_runs), desc="Refinement 1 ...", disable=not verbose):
            sample_points = A[
                rng.choice(
                    len(A),
                    size=int(
                        subsample * len(A)
                    ),  # Use the same subsample percentage as before
                    replace=False,
                )
            ]
            _, neighbors = nn.kneighbors(normalize(sample_points @ self.W))
            W_new, _ = orthogonal_procrustes(sample_points, B[neighbors].mean(axis=1))

            self.W = 0.5 * self.W + 0.5 * W_new

        # --- Refinement 2: Cluter-Based Alignment Correction ---
        kmeans1 = KMeans(
            n_clusters=n_clusters, random_state=rng.integers(1_000_000)
        ).fit(A)
        centers1 = kmeans1.cluster_centers_
        kmeans2 = KMeans(
            n_clusters=n_clusters,
            random_state=rng.integers(1_000_000),
            init=centers1 @ self.W,
        ).fit(B)
        W_new, _ = orthogonal_procrustes(centers1, kmeans2.cluster_centers_)
        self.W = 0.5 * self.W + 0.5 * W_new
        return self

    # Helper functions
    def _align_centroids(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int,
        n_runs: int,
        subsample: float,
        rng: numpy.random.default_rng,
    ):
        """
        Create the clusters centers from a subsample of the input data.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_clusters (int):
                Number of clusters
            n_runs (int):
                Number of runs to try and find the optimal alignment
            subsample (float):
                Percentage of data to use as a subsample to cluster
            rng (numpy.random.default_rng):
                Numpy random number generator

        Returns: (numpy.array, numpy.array)
            Cluster centers for subsets of A and B respectively
        """
        # Use subsample and K-Means
        A_clusterer = KMeans(n_clusters=n_clusters).fit(
            A[rng.choice(len(A), size=int(subsample * len(A)), replace=False)]
        )
        A_clusters = normalize(A_clusterer.cluster_centers_)

        B_clusterer = KMeans(n_clusters=n_clusters).fit(
            B[rng.choice(len(B), size=int(subsample * len(B)), replace=False)]
        )
        B_clusters = normalize(B_clusterer.cluster_centers_)

        quad = None
        # need to re-run the QAP a few times because it's not very good at finding the global optimum (even 2opt)
        for _ in range(n_runs):
            new_quad = quadratic_assignment(
                A_clusters @ A_clusters.T,
                B_clusters @ B_clusters.T,
                method="2opt",
                options={"maximize": True},
            )
            # TODO: Make method and options configurable
            if quad is None or quad.fun < new_quad.fun:
                quad = new_quad

        return A_clusters, B_clusters[quad.col_ind]

    def transform(self, X: numpy.array):
        """
        Transform the input embeddings `X`.

        Args:
            X (numpy.array):
                Set of embeddings to transform

        Returns: numpy.array
            Transformed embeddings X

        """
        if self.W is None:
            raise RuntimeError("Run `.fit` first")

        # Normalize X
        X_mean = X.mean(axis=0)
        X = normalize(X - X_mean)
        return X @ self.W

    def fit_transform(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int = 20,
        n_runs: int = 30,
        top_k: int = 50,
        subsample: float = 0.33,
        random_seed: int = 123,
        verbose: bool = True,
    ):
        """
        Create the optimal matrix `W` of linear transforms for
        mapping embeddings from embedding space A to embedding
        space B. Then apply it to the embeddings from A.
        Same as running `.fit().transform(X)`.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_clusters (int, Optional):
                Number of clusters for k-means for finding centroids
                Default is 20

            n_runs (int, Optional):
                Number of iterations to run to get a varied set of
                centers that represent the embedding spaces.
                Default is 30

            top_k (int, Optional):
                TODO: Understand

            subsample (float or None, Optional):
                For each iteration use a percentage subset of the
                data to in A and B, to get the centers.
                Value must be Between 0 and 1.
                If 1 use all the data.
                Default is 0.33

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Default is 123

            verbose (bool, Optional):
                If False it will hide all the progress bars
                Default is True

        Returns: self
            Fitted MiniVec2Vec object
        """
        return self.fit(
            A, B, n_clusters, n_runs, top_k, subsample, random_seed, verbose
        ).transform(A)
