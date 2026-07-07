import unittest

import numpy as np

from app.services.corpus.topics import (
    _agglomerative_single_link_threshold,
    _assign_noise_to_nearest_cluster,
    _cluster_topic_embeddings,
)
from app.utils.vectors import l2_normalize


class TopicsClusteringTests(unittest.TestCase):
    def test_agglomerative_handles_uneven_cluster_sizes(self) -> None:
        rng = np.random.RandomState(7)
        c1 = np.array([1.0, 0.0, 0.0], dtype="float32")
        c2 = np.array([0.0, 1.0, 0.0], dtype="float32")
        c3 = np.array([0.0, 0.0, 1.0], dtype="float32")

        g1 = c1 + 0.01 * rng.randn(20, 3).astype("float32")
        g2 = c2 + 0.01 * rng.randn(7, 3).astype("float32")
        g3 = c3 + 0.01 * rng.randn(3, 3).astype("float32")
        X = l2_normalize(np.vstack([g1, g2, g3]))

        labels = _agglomerative_single_link_threshold(X, threshold=0.92)
        counts = sorted([int(np.sum(labels == cid)) for cid in set(labels.tolist())], reverse=True)
        self.assertEqual(counts, [20, 7, 3])

    def test_noise_assignment_keeps_all_points_clustered(self) -> None:
        X = l2_normalize(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.98, 0.02, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.99, 0.01],
                    [0.62, 0.60, 0.0],  # "noise-like" point
                ],
                dtype="float32",
            )
        )
        labels = np.array([0, 0, 1, 1, -1], dtype="int64")
        reassigned = _assign_noise_to_nearest_cluster(X, labels)
        self.assertTrue(np.all(reassigned >= 0))
        self.assertEqual(reassigned.shape[0], X.shape[0])

    def test_cluster_topic_embeddings_explicit_kmeans_mode(self) -> None:
        rng = np.random.RandomState(13)
        a = np.array([1.0, 0.0], dtype="float32") + 0.02 * rng.randn(10, 2).astype("float32")
        b = np.array([0.0, 1.0], dtype="float32") + 0.02 * rng.randn(10, 2).astype("float32")
        X = np.vstack([a, b]).astype("float32")

        labels, meta = _cluster_topic_embeddings(X=X, seed=0, algorithm="kmeans")
        self.assertEqual(labels.shape[0], X.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertEqual(meta.get("method"), "kmeans_embeddings")
        self.assertIsNotNone(meta.get("k"))

    def test_cluster_topic_embeddings_auto_assigns_all_points(self) -> None:
        rng = np.random.RandomState(21)
        c1 = np.array([1.0, 0.0, 0.0], dtype="float32")
        c2 = np.array([0.0, 1.0, 0.0], dtype="float32")
        c3 = np.array([0.0, 0.0, 1.0], dtype="float32")
        g1 = c1 + 0.03 * rng.randn(18, 3).astype("float32")
        g2 = c2 + 0.03 * rng.randn(9, 3).astype("float32")
        g3 = c3 + 0.03 * rng.randn(4, 3).astype("float32")
        X = np.vstack([g1, g2, g3]).astype("float32")

        labels, meta = _cluster_topic_embeddings(X=X, seed=0, algorithm="auto")
        self.assertEqual(labels.shape[0], X.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertIn(
            meta.get("method"),
            {"hdbscan_embeddings", "agglomerative_cosine_threshold", "kmeans_embeddings"},
        )


if __name__ == "__main__":
    unittest.main()
