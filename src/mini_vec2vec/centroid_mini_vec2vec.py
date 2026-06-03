import numpy
from tqdm.auto import tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from .mini_vec2vec_base import MiniVec2VecBase


class CentroidMiniVec2Vec(MiniVec2VecBase):
    def __init__(self):
        """
        Constructor for Mini-Vec2Vec using the centroids only
        method. (Option 2 in the paper)
        """
        self.W = None

    def fit(
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
        mapping embeddings from embedding space `A` to embedding
        space `B`, using relative representations.

        This provides an easier to use but less configurable
        implementation of the paper. If you want a more configurable
        implementation I would recommend you run `match_anchors`,
        `refinement_1`, and `refinement_2` separately.

        Args:
            A (numpy.array):
                Embeddings from the embedding space you want to transform from.

            B (numpy.array):
                Embeddings from the embedding space you want to transform to.

            n_clusters (int, Optional):
                Number of clusters for k-means for finding centroids for both
                the `matching_anchors` and `refinement_2` steps
                Default is 20

            n_runs (int, Optional):
                Number of iterations to run to get a varied set of centers that
                represent the embedding spaces. Used in both `matching_anchors`
                and `refinement_1`.
                Default is 30

            top_k (int, Optional):
                In the relative representation space for embedding in `A` and `B`,
                find the top k nearest neighbors from relative space from `B` for each `A`. Only used in `refinement_1`.
                (`A` and `B` share the same representation space)

            alpha (float, Optional):
                Smoothing factor when running exponential smoothing
                for updating the transform matrix `W`.
                Must be `0 < alpha < 1`.
                Used in both `refinement_1` and `refinement_2`.
                Default is 0.5

            subsample (float, Optional):
                For each iteration use a percentage subset of the
                data to in `A` and `B`, to get the centers.
                Must be `0 < subsample <= 1`.
                Set `subsample=1` to use all the data.
                Used in all `matching_anchors`, `refinement_1`, and `refinement_2`.
                Default is 0.33

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Used in all `matching_anchors`, `refinement_1`, and `refinement_2`.
                Default is 123

            verbose (bool, Optional):
                If `False` it will hide all the progress bars
                Used in `matching_anchors` and `refinement_1`.
                Default is True

        Returns:
            self (CentroidMiniVec2Vec):
                Fitted MiniVec2Vec object
        """
        return (
            self.match_anchors(
                A, B, n_clusters, n_runs, subsample, random_seed, verbose
            )
            .refinement_1(
                A,
                B,
                n_runs,
                top_k,
                subsample,
                alpha,
                random_seed,
                verbose,
            )
            .refinement_2(
                A,
                B,
                n_clusters,
                alpha,
                random_seed,
            )
        )

    def match_anchors(
        self,
        A: numpy.array,
        B: numpy.array,
        n_clusters: int = 20,
        n_runs: int = 30,
        subsample: float = 0.33,
        random_seed: int = 123,
        verbose: bool = True,
    ):
        """
        Find an approximate matching between embedding in `A` and `B` using
        centroids.

        1. Perform k-means clustering in each embedding space independently
           and obtain cluster centroids.
        2. Compute pairwise similarities in each space.
        3. Find the optimal matching between the cluster centroids by solving
           the Quadratic Assignment Problem (QAP), which finds a permutation
           that aligns the similarity matrices optimally.
        4. We construct pseudo-parallel pairs by sending each element in space
           `A` to the average of its k nearest neighbors from space `B` (based on
           similarity in relative space).
        5. Optimal orthogonal transformation is obtained by Procrustes analysis

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

            subsample (float or None, Optional):
                For each iteration use a percentage subset of the
                data to in `A` and `B`, to get the centers.
                Must be `0 < subsample <= 1`.
                Set `subsample=1` to use all the data.
                Default is 0.33

            random_seed (int, Optional):
                Use a random seed for reproducible results.
                Default is 123

            verbose (bool, Optional):
                If False it will hide all the progress bars
                Default is True

        Returns:
            self (CentroidMiniVec2Vec):
                MiniVec2Vec object after match anchors have been applied
        """
        if not 0 < subsample <= 1:
            raise ValueError("`subsample` must be: 0 < `subsample` <= 1")

        # Set up
        A, B = self._pre_process_embeddings(A, B)
        rng = numpy.random.default_rng(random_seed)

        # get centroids
        A_centers = []
        B_centers = []
        for _ in tqdm(range(n_runs), desc="Matching Anchors ...", disable=not verbose):
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
                if quad is None or quad.fun < new_quad.fun:
                    quad = new_quad

            A_centers.append(A_clusters)
            B_centers.append(B_clusters[quad.col_ind])

        # Run procrustes over the aligned centroids
        r_A = numpy.vstack(A_centers)
        r_B = numpy.vstack(B_centers)

        # --- Train Mappings ---
        self.W, _ = orthogonal_procrustes(r_A, r_B)
        return self
