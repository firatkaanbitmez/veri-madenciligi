import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

def Parametreler():
    """
    Veri seti parametreleri
    """
    parametreler = {
        "n_samples": 650,  # Örnek sayısı
        "n_features": 2,   # Özellik sayısı
        "n_informative": 2,  # Bilgilendirici özellik sayısı
        "n_redundant": 0,   # Gereksiz özellik sayısı
        "n_clusters_per_class": 1,  # Sınıf başına küme sayısı
        "random_state": 10,  # Rastgele durum
    }
    return parametreler

def SNN(X, k_snn, threshold, num_clusters=2):
    """
    Paylaşılan Komşular (SNN) algoritması ile kümeleme
    """
    # Her bir örneğin k-en yakın komşularını hesaplayın
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            distances[i, j] = np.linalg.norm(X[i] - X[j])
            distances[j, i] = distances[i, j]

    # Benzer komşulara sahip örnekleri birleştiren bir grafik oluşturun
    graph = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        neighbors = np.argsort(distances[i])[:k_snn]
        for j in neighbors:
            if distances[i, j] <= threshold:
                graph[i, j] = 1
                graph[j, i] = 1

    # Grafik kullanarak kümeleme yapın
    clusters = []
    visited = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        if not visited[i]:
            cluster = []
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    cluster.append(node)
                    for j in range(X.shape[0]):
                        if graph[node, j] == 1 and not visited[j]:
                            stack.append(j)
            clusters.append(cluster)

    # Bulunan küme sayısı num_clusters'dan fazlaysa birleştirin
    while len(clusters) > num_clusters:
        min_index = np.argmin([len(cluster) for cluster in clusters])
        min_cluster = clusters.pop(min_index)
        for point in min_cluster:
            closest_cluster_idx = np.argmin([np.inf if cluster == min_cluster else min(distances[point, cluster_point] for cluster_point in cluster) for cluster in clusters])
            clusters[closest_cluster_idx].append(point)

    return clusters

def BenimKMeans(X, k_means):
    """
    K-means kümeleme algoritması
    """
    kmeans = KMeans(n_clusters=k_means, random_state=10)
    kmeans.fit(X)
    return kmeans.labels_

def plot_clusters(X, labels, title, axs):
    """
    Kümeleme sonuçlarını görselleştir
    """
    axs.scatter(X[:, 0], X[:, 1], c=labels)
    axs.set_title(title)

def main():
    parameters = Parametreler()
    X, y = make_classification(**parameters)

    # SNN kümeleme
    k_snn = 2
    threshold = 0.8  # Eşik değeri
    num_clusters = 2  # Bölmek istediğiniz küme sayısı
    snn_clusters = SNN(X, k_snn, threshold, num_clusters=num_clusters)

    # Etiketler listesini oluşturun
    snn_labels = np.zeros(X.shape[0], dtype=int)
    for cluster_id, cluster in enumerate(snn_clusters):
        for point_idx in cluster:
            snn_labels[point_idx] = cluster_id

    # K-means kümeleme
    k_means = 2
    kmeans_clusters = BenimKMeans(X, k_means)

    # Grafikleri yan yana göstermek için subplot kullanımı
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    plot_clusters(X, snn_labels, "SNN Kümeleme", axs[0])
    plot_clusters(X, kmeans_clusters, "K-means Kümeleme", axs[1])

    # Performans karşılaştırması
    snn_accuracy = np.mean(y == snn_labels)
    kmeans_accuracy = np.mean(y == kmeans_clusters)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
