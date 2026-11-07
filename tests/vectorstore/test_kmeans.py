import numpy as np
import pytest

from rag_agent.vectorstore.kmeans import assign, kmeans, squared_distances


def make_blobs(n_per_cluster=50, centers=((0, 0), (10, 10), (-10, 10)), spread=0.5, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=np.float64)
    points = np.vstack([c + rng.normal(scale=spread, size=(n_per_cluster, centers.shape[1])) for c in centers])
    truth = np.repeat(np.arange(len(centers)), n_per_cluster)
    return points, truth, centers


def same_partition(labels_a, labels_b):
    """True if both labelings group the points identically (up to cluster renaming)."""
    pairs = set(zip(labels_a.tolist(), labels_b.tolist()))
    return len(pairs) == len(set(labels_a.tolist())) == len(set(labels_b.tolist()))


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_recovers_well_separated_clusters(init):
    points, truth, centers = make_blobs()

    result = kmeans(points, n_clusters=3, init=init, seed=0)

    assert result.centroids.shape == (3, 2)
    assert result.labels.shape == (len(points),)
    assert same_partition(result.labels, truth)
    # each true center has a centroid right next to it
    nearest = np.sqrt(squared_distances(centers, result.centroids)).min(axis=1)
    assert (nearest < 0.3).all()
    assert result.converged


def test_inertia_never_increases_across_iterations():
    points, _, _ = make_blobs(n_per_cluster=40, centers=[(i * 4, (i % 3) * 4) for i in range(6)], spread=1.5)

    inertias = [kmeans(points, n_clusters=6, max_iter=n, tol=0.0, seed=0).inertia for n in range(1, 15)]

    assert all(later <= earlier + 1e-9 for earlier, later in zip(inertias, inertias[1:]))


def test_converged_centroids_are_the_mean_of_their_cluster():
    points, _, _ = make_blobs(n_per_cluster=60, centers=[(0, 0), (6, 0), (3, 5), (-4, 6)], spread=1.0, seed=2)

    result = kmeans(points, n_clusters=4, tol=1e-9, seed=0)

    assert result.converged
    assert result.n_iter < 100
    for cluster, centroid in enumerate(result.centroids):
        np.testing.assert_allclose(centroid, points[result.labels == cluster].mean(axis=0), atol=1e-6)


def test_labels_point_to_the_nearest_centroid():
    points = np.random.default_rng(3).normal(size=(200, 5))

    result = kmeans(points, n_clusters=7, seed=1)

    expected = np.linalg.norm(points[:, None, :] - result.centroids[None], axis=2).argmin(axis=1)
    np.testing.assert_array_equal(result.labels, expected)
    assert result.inertia == pytest.approx(((points - result.centroids[result.labels]) ** 2).sum())


def test_is_deterministic_for_a_given_seed():
    points = np.random.default_rng(4).normal(size=(300, 8))

    a = kmeans(points, n_clusters=10, seed=42)
    b = kmeans(points, n_clusters=10, seed=42)

    np.testing.assert_array_equal(a.centroids, b.centroids)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_stops_at_max_iter_when_not_converged():
    points = np.random.default_rng(5).normal(size=(500, 16))

    result = kmeans(points, n_clusters=20, max_iter=2, tol=0.0, seed=0)

    assert result.n_iter == 2
    assert not result.converged


def test_kmeans_plus_plus_beats_or_matches_random_init_on_average():
    points, _, _ = make_blobs(n_per_cluster=40, centers=[(i * 5, 0) for i in range(8)], spread=0.8)

    plus = np.mean([kmeans(points, 8, init="k-means++", max_iter=1, seed=s).inertia for s in range(10)])
    rand = np.mean([kmeans(points, 8, init="random", max_iter=1, seed=s).inertia for s in range(10)])

    assert plus <= rand


def test_every_cluster_is_non_empty_even_with_duplicate_points():
    # 3 distinct locations, heavily duplicated: forces empty clusters at some point
    points = np.repeat(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 20, axis=0)
    points = np.vstack([points, [[5.0, 5.0], [6.0, 5.0]]])

    result = kmeans(points, n_clusters=5, init="random", seed=0)

    assert np.bincount(result.labels, minlength=5).min() >= 1


def test_as_many_clusters_as_points_gives_zero_inertia():
    points = np.random.default_rng(6).normal(size=(6, 3))

    result = kmeans(points, n_clusters=6, seed=0)

    assert result.inertia == pytest.approx(0.0, abs=1e-12)


def test_assign_matches_squared_distances():
    rng = np.random.default_rng(7)
    points, centroids = rng.normal(size=(40, 4)), rng.normal(size=(5, 4))

    labels, dists = assign(points, centroids)

    full = ((points[:, None, :] - centroids[None]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(labels, full.argmin(axis=1))
    np.testing.assert_allclose(dists, full.min(axis=1), atol=1e-9)


def test_rejects_invalid_arguments():
    points = np.ones((4, 2))

    with pytest.raises(ValueError, match="n_clusters must be"):
        kmeans(points, n_clusters=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        kmeans(points, n_clusters=5)
    with pytest.raises(ValueError, match="init must be"):
        kmeans(points, n_clusters=2, init="forgy")
    with pytest.raises(ValueError, match="2-D"):
        kmeans(np.ones(4), n_clusters=1)
    with pytest.raises(ValueError, match="max_iter"):
        kmeans(points, n_clusters=1, max_iter=0)
