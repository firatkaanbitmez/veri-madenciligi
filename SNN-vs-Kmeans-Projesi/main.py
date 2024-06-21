import numpy as np
import matplotlib.pyplot as plt

# Veri seti oluşturma
np.random.seed(42)

# İki küme oluşturma
kume1_ort = [1, 1]
kume2_ort = [5, 4]
kume_std = 1.0

veri_kume1 = np.random.randn(100, 2) * kume_std + kume1_ort
veri_kume2 = np.random.randn(100, 2) * kume_std + kume2_ort

veri = np.vstack((veri_kume1, veri_kume2))
etiketler = np.hstack((np.zeros(100), np.ones(100)))  # Küme etiketleri

# Veri setini görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(veri[:, 0], veri[:, 1], c=etiketler, cmap='viridis', marker='o', edgecolors='k')
plt.title('Oluşturulan İki Boyutlu Veri Seti')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.colorbar()
plt.grid(True)
plt.show()

# Shared Nearest Neighbor (SNN) Algoritması
class SNN:
    def __init__(self, eps=1.0, min_pts=5):
        self.eps = eps  # Komşuluk eşiği
        self.min_pts = min_pts  # Minimum komşu sayısı

    def fit(self, veri):
        self.veri = veri
        self.n = len(veri)
        self.etiketler = np.full(self.n, -1)  # -1: Noise, 0, 1, 2, ...: Küme etiketleri
        self.kume_sayisi = 0

        for i in range(self.n):
            if self.etiketler[i] == -1:  # Daha önce etiketlenmemişse
                self.expand_cluster(i)

    def expand_cluster(self, nokta_index):
        komsular = self.get_shared_neighbors(nokta_index)
        
        # Check if the point has enough neighbors to form a cluster
        if len(komsular) < self.min_pts:
            self.etiketler[nokta_index] = -1  # Noise
            return False
        else:
            self.kume_sayisi += 1
            self.etiketler[nokta_index] = self.kume_sayisi
            
            # Iterate over neighbors to assign cluster labels
            for komsu in komsular:
                if self.etiketler[komsu] == -1:  # If neighbor is noise
                    self.etiketler[komsu] = self.kume_sayisi
                elif self.etiketler[komsu] == 0:  # If neighbor is not assigned
                    self.etiketler[komsu] = self.kume_sayisi
                    self.expand_cluster(komsu)  # Recursively expand the cluster
            return True

    def get_shared_neighbors(self, nokta_index):
        komsular = []
        for i in range(self.n):
            if i != nokta_index:
                dist = np.linalg.norm(self.veri[nokta_index] - self.veri[i])
                if dist < self.eps:
                    komsular.append(i)
        return komsular


# K-means Algoritması
class KMeans:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, veri):
        self.veri = veri
        self.n = veri.shape[0]
        self.m = veri.shape[1]
        
        # Rastgele k merkezi seçme
        self.merkezler = veri[np.random.choice(self.n, self.k, replace=False)]
        
        for _ in range(self.max_iter):
            self.etiketler = self.kumelere_ata()
            yeni_merkezler = self.merkezleri_guncelle()
            if np.allclose(self.merkezler, yeni_merkezler):  # Kümelerin merkezleri değişmediyse
                break
            self.merkezler = yeni_merkezler

    def kumelere_ata(self):
        mesafeler = np.zeros((self.n, self.k))
        for i in range(self.k):
            mesafeler[:, i] = np.linalg.norm(self.veri - self.merkezler[i], axis=1)
        return np.argmin(mesafeler, axis=1)

    def merkezleri_guncelle(self):
        yeni_merkezler = np.zeros((self.k, self.m))
        for i in range(self.k):
            yeni_merkezler[i] = np.mean(self.veri[self.etiketler == i], axis=0)
        return yeni_merkezler

# SNN ve K-means uygulaması
snn = SNN(eps=0.7, min_pts=5)
snn.fit(veri)
snn_etiketler = snn.etiketler

kmeans = KMeans(k=2, max_iter=100)
kmeans.fit(veri)
kmeans_etiketler = kmeans.etiketler

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))

# SNN sonuçları
plt.subplot(1, 2, 1)
plt.scatter(veri[:, 0], veri[:, 1], c=snn_etiketler, cmap='viridis', marker='o', edgecolors='k')
plt.title('SNN Kümeleme Sonuçları')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')

# K-means sonuçları
plt.subplot(1, 2, 2)
plt.scatter(veri[:, 0], veri[:, 1], c=kmeans_etiketler, cmap='viridis', marker='o', edgecolors='k')
plt.scatter(kmeans.merkezler[:, 0], kmeans.merkezler[:, 1], s=300, c='red', marker='X')
plt.title('K-means Kümeleme Sonuçları')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')

plt.tight_layout()
plt.show()
