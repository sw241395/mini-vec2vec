import pytest
import numpy
from mini_vec2vec import MiniVec2Vec, CentroidMiniVec2Vec
from sklearn.metrics.pairwise import cosine_similarity


class BaseTestMiniVec2Vec:
    def test_fit(self, mv2v, A, B):
        # mv2v = MiniVec2Vec()
        mv2v.fit(A, B, n_clusters=3, n_runs=3, top_k=3, verbose=False)
        assert mv2v.W.shape == (A.shape[1], B.shape[1])

    def test_fit_transform(self, mv2v, A, B):
        transformed_a = mv2v.fit_transform(
            A, B, n_clusters=3, n_runs=3, top_k=3, verbose=False
        )
        # Get cosine sim between transformed A and B
        cosine_sim1 = [
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            for a, b in zip(transformed_a, B)
        ]
        # Get cosine sim between transformed A and A
        cosine_sim2 = [
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            for a, b in zip(transformed_a, A)
        ]
        # Test to see the transformed A should be more similar to B than A
        assert sum(
            numpy.argmax([x, y]).item() for x, y in zip(cosine_sim1, cosine_sim2)
        ) <= len(A)

    def test_transform_no_fit(self, mv2v):
        """Test the method will error when fit is not called first"""
        with pytest.raises(RuntimeError):
            mv2v.transform(numpy.ones((3, 4)))

    def test_fit_and_transform(self, mv2v, A, B):
        # mv2v = MiniVec2Vec()
        mv2v.fit(A, B, n_clusters=3, n_runs=3, top_k=3, verbose=False)
        transformed_a = mv2v.transform(A)
        # Get cosine sim between transformed A and B
        cosine_sim1 = [
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            for a, b in zip(transformed_a, B)
        ]
        # Get cosine sim between transformed A and A
        cosine_sim2 = [
            cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            for a, b in zip(transformed_a, A)
        ]
        # Test to see the transformed A should be more similar to B than A
        assert sum(
            numpy.argmax([x, y]).item() for x, y in zip(cosine_sim1, cosine_sim2)
        ) <= len(A)

    @pytest.mark.parametrize("to_cut", ["A", "B"])
    def test_different_embedding_dims(self, mv2v, A, B, to_cut):
        if to_cut == "A":
            test_A = A[:, : A.shape[1] // 2]
            test_B = B
        elif to_cut == "B":
            test_A = A
            test_B = B[:, : B.shape[1] // 2]
        else:
            raise ValueError(f"`to_cut` var {to_cut} not valid")

        mv2v.fit(test_A, test_B, n_clusters=3, n_runs=3, top_k=3, verbose=False)
        # Should have no issue fitting different sized embeddings
        assert mv2v.W.shape == (A.shape[1], B.shape[1])


class TestMiniVec2Vec(BaseTestMiniVec2Vec):
    @pytest.fixture(scope="function")
    def mv2v(self):
        return MiniVec2Vec()


class TestCentroidMiniVec2Vec(BaseTestMiniVec2Vec):
    @pytest.fixture(scope="function")
    def mv2v(self):
        return CentroidMiniVec2Vec()
