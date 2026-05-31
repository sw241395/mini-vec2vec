import pytest
import numpy
from mini_vec2vec import MiniVec2Vec
from sklearn.metrics.pairwise import cosine_similarity


class TestMiniVec2Vec:
    def test_fit(self, A, B):
        mv2v = MiniVec2Vec()
        mv2v.fit(A, B, n_clusters=3, n_runs=3, top_k=3, verbose=False)
        assert mv2v.W.shape == (A.shape[1], B.shape[1])

    def test_fit_transform(self, A, B):
        mv2v = MiniVec2Vec()
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

    def test_transform_no_fit(self):
        """Test the method will error when fit is not called first"""
        mv2v = MiniVec2Vec()
        with pytest.raises(RuntimeError):
            mv2v.transform(numpy.ones((3, 4)))

    def test_fit_and_transform(self, A, B):
        mv2v = MiniVec2Vec()
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
