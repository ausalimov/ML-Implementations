from time import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import numpy as np 
from kmeans import RegularKMeans

# Taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

class RandomEstimator():
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.inertia_ = 0 
    
    def fit(self, X): 
        self.labels_ = np.random.randint(0, self.n_clusters, size=len(X))
        return self

def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    X = StandardScaler().fit_transform(data)
    estimator = kmeans.fit(X)
    fit_time = time() - t0
    results = [name, fit_time, estimator.inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator.labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator.labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


if __name__ == '__main__': 

    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
    print("\n\n\n")


    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=1)
    bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

    kmeans = KMeans(init="random", n_clusters=n_digits, n_init=1)
    bench_k_means(kmeans=kmeans, name="random_points", data=data, labels=labels)

    kmeans = RegularKMeans(init="kpp", n_clusters=n_digits)
    bench_k_means(kmeans=kmeans, name="custom", data=data, labels=labels)

    random_estimator = RandomEstimator(n_clusters=n_digits)
    bench_k_means(kmeans=random_estimator, name="null_model", data=data, labels=labels)

    print(82 * "_")

