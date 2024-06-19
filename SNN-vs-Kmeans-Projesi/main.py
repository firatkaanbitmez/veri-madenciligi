import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Veri seti oluşturma
np.random.seed(42)
data1 = np.random.randn(100, 2) + np.array([5, 5])
data2 = np.random.randn(100, 2) + np.array([0, 0])
data = np.vstack((data1, data2))

# Veri görselleştirme
plt.scatter(data[:, 0], data[:, 1])
plt.title("Oluşturulan Veri Seti")
plt.show()

# Veri normalizasyonu
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

normalized_data = normalize(data)

# SNN (Simple Nearest Neighbor) algoritması
class SNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, data):
        self.data = data
        self.dist_matrix = distance_matrix(data, data)
        self.similarity_matrix = np.zeros(self.dist_matrix.shape)

        for i in range(len(data)):
            neighbors = np.argsort(self.dist_matrix[i])[:self.k+1]
            for j in neighbors:
                self.similarity_matrix[i, j] = 1

    def predict(self, point):
        dist = np.linalg.norm(self.data - point, axis=1)
        neighbors = np.argsort(dist)[:self.k]
        return np.sum(self.similarity_matrix[neighbors], axis=0)

# K-means algoritması
class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, data):
        self.data = data
        self.centroids = data[np.random.choice(len(data), self.k, replace=False)]
        
        for _ in range(self.max_iters):
            self.labels = self.assign_clusters()
            new_centroids = self.calculate_centroids()
            
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids

    def assign_clusters(self):
        distances = np.array([np.linalg.norm(self.data - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def calculate_centroids(self):
        return np.array([self.data[self.labels == i].mean(axis=0) for i in range(self.k)])

# SNN ve K-means algoritmalarını uygulama
snn = SNN(k=3)
snn.fit(normalized_data)

kmeans = KMeans(k=2)
kmeans.fit(normalized_data)

# K-means sonuçlarını görselleştirme
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=kmeans.labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X')
plt.title("K-means Sonuçları")
plt.show()

# SNN sonuçları için örnek tahmin
point = np.array([0, 0])
snn_result = snn.predict(point)
print(f"SNN Tahmini: {snn_result}")
