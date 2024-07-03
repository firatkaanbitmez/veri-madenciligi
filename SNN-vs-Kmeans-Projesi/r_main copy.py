import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# Parameters function
def Parametreler():
    parametreler = {
        "samples": 650,
        "n_features": 2,
        "n_informative": 2,
        "n_redundant": 0,
        "n_clusters_per_class": 1,
        "random_state": 10,
        "snn_eps": 0.25,        # Shared Nearest Neighbor parameters
        "snn_min_pts": 5,       # Adjusted min_pts value
        "kmeans_k": 2,
        "kmeans_max_iter": 100
    }
    return parametreler

# Shared Nearest Neighbor (SNN) Algorithm
class SNN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts

    def fit(self, X):
        self.X = X
        self.nbrs = NearestNeighbors(radius=self.eps).fit(X)
        self.snn_matrix = self.compute_snn_matrix()
        self.labels = self.cluster_points()

    def compute_snn_matrix(self):
        distances, indices = self.nbrs.radius_neighbors(self.X)
        snn_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                if i != neighbor:
                    snn_matrix[i][neighbor] += 1
                    snn_matrix[neighbor][i] += 1  # symmetric matrix
        return snn_matrix

    def cluster_points(self):
        clusters = np.full(self.X.shape[0], -1)
        cluster_id = 0
        for i in range(self.X.shape[0]):
            if clusters[i] == -1:
                neighbors = np.where(self.snn_matrix[i] >= self.min_pts)[0]
                if len(neighbors) >= self.min_pts:
                    clusters[i] = cluster_id
                    clusters = self.expand_cluster(clusters, i, neighbors, cluster_id)
                    cluster_id += 1
        return clusters

    def expand_cluster(self, clusters, point_id, neighbors, cluster_id):
        queue = list(neighbors)
        while queue:
            current_point = queue.pop(0)
            if clusters[current_point] == -1:
                clusters[current_point] = cluster_id
                current_neighbors = np.where(self.snn_matrix[current_point] >= self.min_pts)[0]
                if len(current_neighbors) >= self.min_pts:
                    queue.extend(current_neighbors)
        return clusters

# Main function
def main():
    # Load parameters
    parametreler = Parametreler()

    # Create dataset
    X, _ = make_classification(n_samples=parametreler["samples"],
                               n_features=parametreler["n_features"],
                               n_informative=parametreler["n_informative"],
                               n_redundant=parametreler["n_redundant"],
                               n_clusters_per_class=parametreler["n_clusters_per_class"],
                               random_state=parametreler["random_state"])

    # Apply SNN and K-means algorithms
    snn = SNN(eps=parametreler["snn_eps"], min_pts=parametreler["snn_min_pts"])
    snn.fit(X)
    snn_labels = snn.labels

    kmeans = KMeans(n_clusters=parametreler["kmeans_k"], max_iter=parametreler["kmeans_max_iter"])
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_

    # Visualize results
    plt.figure(figsize=(12, 6))

    # SNN results
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=snn_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title('SNN Clustering Results')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    # K-means results
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    plt.title('K-means Clustering Results')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    plt.tight_layout()
    plt.show()

    # Print SNN and K-means results in the terminal
    print("SNN Clustering Results:")
    print(snn_labels)

    print("\nK-means Clustering Results:")
    print(kmeans_labels)

if __name__ == "__main__":
    main()
