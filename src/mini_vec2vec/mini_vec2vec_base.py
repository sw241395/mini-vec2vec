import numpy
import warnings
from tqdm.auto import tqdm
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class MiniVec2VecBase(ABC):
    @abstractmethod
    def fit(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int,
        n_runs: int,
        top_k: int,
        alpha: float,
        subsample: float,
        random_seed: int,
        verbose: bool,
    ):
        raise NotImplementedError()

    @abstractmethod
    def match_anchors(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int,
        n_runs: int,
        top_k: int,
        subsample: float,
        random_seed: int,
        verbose: bool,
    ):
        raise NotImplementedError()

    def refinement_1(
        self,
        A: numpy.array,
        B: numpy.array,
        n_runs: int = 30,
        top_k: int = 50,
        subsample: float = 0.33,
        alpha: float = 0.5,
        random_seed: int = 123,
        verbose: bool = True,
    ):
        """
        Refinement 1: Iterative Closest Point Average
        Transforming embed-dings from space A to B using W,
        averaging their nearest neighbors in space B, and
        obtaining a new orthogonal transformation with
        Procrustes analysis.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_runs (int, Optional):
                Number of iterations to run to get a varied set of
                centers that represent the embedding spaces.
                Default is 30

            top_k (int, Optional):
                In the relative representation space for embedding in A and B,
                find the top k nearest neighbors from relative space from B for each A.
                (A and B share the same representation space)

            subsample (float, Optional):
                For each iteration of refinement 1, use a subset of embeddings
                in A. To reduce overhead and reduce overfitting.
                Default is 0.33

            alpha (float, Optional):
                Smoothing factor when running exponential smoothing
                for updating the transform matrix W.
                Must be 0 < alpha < 1.
                Default is 0.5

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Default is 123

            verbose (bool, Optional):
                If False it will hide all the progress bars
                Default is True

        Returns: self
            MiniVec2Vec object after refinement 1 have been applied
        """
        if not 0 < subsample <= 1:
            raise ValueError("`subsample` must be: 0 < `subsample` <= 1")
        if not 0 < alpha < 1:
            raise ValueError("`alpha` must be: 0 < `alpha` < 1")

        # Set up
        self._check_w()
        A, B = self._pre_process_embeddings(A, B)
        rng = numpy.random.default_rng(random_seed)

        # Fit KNN on B
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
            self.W = (1 - alpha) * self.W + alpha * W_new
        return self

    def refinement_2(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int = 50,
        alpha: float = 0.5,
        random_seed: int = 123,
    ):
        """
        Refinement 2: Cluter-Based Alignment Correction
        Improve the large-scale matching between the spaces, by
        clustering the embeddings in space A. We then apply W to
        the cluster centroids and cluster the B embeddings, where
        the clustering algorithm is initialized with the transformed
        A centroids as the initial centroids. The transformed
        clusters should be close to a set of clusters in the new
        space, therefore expect the clustering algorithm to make only
        minor adjustments, correcting biases in the transformation and
        moving the centroids to their “right positions” in space B.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_clusters (int, Optional):
                Number of clusters for k-means for finding centroids
                Default is 20

            alpha (float, Optional):
                Smoothing factor when running exponential smoothing
                for updating the transform matrix W.
                Must be 0 < alpha < 1.
                Default is 0.5

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Default is 123

        Returns: self
            MiniVec2Vec object after refinement 2 have been applied
        """
        if not 0 < alpha < 1:
            raise ValueError("`alpha` must be: 0 < `alpha` < 1")

        # Set up
        self._check_w()
        A, B = self._pre_process_embeddings(A, B)
        rng = numpy.random.default_rng(random_seed)

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
        self.W = (1 - alpha) * self.W + alpha * W_new
        return self

    def transform(self, X: numpy.array):
        """
        Transform the input embeddings `X`.

        Args:
            X (numpy.array):
                Set of embeddings to transform

        Returns: numpy.array
            Transformed embeddings X

        """
        self._check_w()

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
        alpha: float = 0.5,
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
            A, B, n_clusters, n_runs, top_k, alpha, subsample, random_seed, verbose
        ).transform(A)

    def _check_w(self):
        """Check self.w has been initialized"""
        if self.W is None:
            raise RuntimeError(
                "Initial match anchors have not been initialized. PLeas run `.fit` or `.match_anchors` first."
            )

    def _pre_process_embeddings(
        self,
        A: numpy.array,
        B: numpy.array,
    ):
        """
        Preprocess embedding A, B
        1. Check dimensions of embeddings
        2. Take away the mean and normalize such that the embeddings
           center around the unit hyper-sphere.
        3. Zero pad smaller embedding to enable working with different
           embedding dimensions.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

        return (numpy.array, numpy.array)
            Preprocess A and B
        """
        # Check input embeddings dims
        assert A.ndim == 2
        assert B.ndim == 2

        # Preprocess embeddings s.t. they center around the mean and normalize
        A = normalize(A - A.mean(axis=0))
        B = normalize(B - B.mean(axis=0))

        if A.shape[1] < B.shape[1]:
            warnings.warn(
                "Zero padding embeddings A to match embedding dimension of embeddings B"
            )
            A = numpy.pad(A, ((0, 0), (0, B.shape[1] - A.shape[1])), "constant")
        elif A.shape[1] > B.shape[1]:
            warnings.warn(
                "Zero padding embeddings B to match embedding dimension of embeddings A"
            )
            B = numpy.pad(B, ((0, 0), (0, A.shape[1] - B.shape[1])), "constant")

        return A, B
