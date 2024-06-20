import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Veri seti oluşturma
np.random.seed(42)
veri_kume1 = np.random.randn(100, 2) + np.array([5, 5])  # 1. küme
veri_kume2 = np.random.randn(100, 2) + np.array([0, 0])  # 2. küme
veri = np.vstack((veri_kume1, veri_kume2))  # İki kümeyi birleştir

# Veri görselleştirme
plt.scatter(veri[:, 0], veri[:, 1])
plt.title("Oluşturulan Veri Seti")
plt.show()

# Veri normalizasyonu
def normalizasyon(veri):
    ortalama = np.mean(veri, axis=0)
    std_sapma = np.std(veri, axis=0)
    return (veri - ortalama) / std_sapma

normalizasyonlu_veri = normalizasyon(veri)

# SNN (Simple Nearest Neighbor) algoritması
class SNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, veri, etiketler):
        self.veri = veri
        self.etiketler = etiketler
        self.mesafe_matrisi = distance_matrix(veri, veri)
        self.benzerlik_matrisi = np.zeros(self.mesafe_matrisi.shape)

        for i in range(len(veri)):
            komsular = np.argsort(self.mesafe_matrisi[i])[:self.k+1]
            for j in komsular:
                self.benzerlik_matrisi[i, j] = 1

    def tahmin(self, nokta):
        mesafe = np.linalg.norm(self.veri - nokta, axis=1)
        komsular = np.argsort(mesafe)[:self.k]
        komsu_etiketler = self.etiketler[komsular]
        etiket_sayilari = np.bincount(komsu_etiketler.astype(int))
        return np.argmax(etiket_sayilari), etiket_sayilari

# K-means algoritması
class KMeans:
    def __init__(self, k=2, max_iterasyon=100):
        self.k = k
        self.max_iterasyon = max_iterasyon

    def fit(self, veri):
        self.veri = veri
        self.merkezler = veri[np.random.choice(len(veri), self.k, replace=False)]
        
        for _ in range(self.max_iterasyon):
            self.etiketler = self.kumeleri_ata()
            yeni_merkezler = self.merkezleri_hesapla()
            
            if np.all(self.merkezler == yeni_merkezler):
                break
            
            self.merkezler = yeni_merkezler

    def kumeleri_ata(self):
        mesafeler = np.array([np.linalg.norm(self.veri - merkez, axis=1) for merkez in self.merkezler])
        return np.argmin(mesafeler, axis=0)

    def merkezleri_hesapla(self):
        return np.array([self.veri[self.etiketler == i].mean(axis=0) for i in range(self.k)])

# Etiketleri oluşturma
etiketler1 = np.zeros(100)  # 1. küme etiketleri
etiketler2 = np.ones(100)   # 2. küme etiketleri
etiketler = np.hstack((etiketler1, etiketler2))

# SNN ve K-means algoritmalarını uygulama
snn = SNN(k=3)
snn.fit(normalizasyonlu_veri, etiketler)

kmeans = KMeans(k=2)
kmeans.fit(normalizasyonlu_veri)

# K-means sonuçlarını görselleştirme
plt.scatter(normalizasyonlu_veri[:, 0], normalizasyonlu_veri[:, 1], c=kmeans.etiketler, cmap='viridis', marker='o')
plt.scatter(kmeans.merkezler[:, 0], kmeans.merkezler[:, 1], s=300, c='red', marker='X')
plt.title("K-means Sonuçları")
plt.show()

# SNN sonuçları için örnek tahmin
tahmin_nokta = np.array([0, 0])
snn_etiket, snn_etiket_sayilari = snn.tahmin(tahmin_nokta)
print(f"SNN Tahmini Etiketi: {snn_etiket}, Etiket Dağılımı: {snn_etiket_sayilari}")

# K-means sonuçlarının terminal çıktısı
print("\nK-means Sonuçları:")
print(f"Merkezler: {kmeans.merkezler}")
print(f"Etiketler: {kmeans.etiketler}")

# Veri seti ve etiketlerin görselleştirilmesi
plt.figure(figsize=(18, 6))

# Orijinal veri seti
plt.subplot(1, 3, 1)
plt.scatter(veri[:, 0], veri[:, 1], c=etiketler, cmap='viridis', marker='o')
plt.title("Orijinal Veri Seti")

# K-means sonuçları
plt.subplot(1, 3, 2)
plt.scatter(normalizasyonlu_veri[:, 0], normalizasyonlu_veri[:, 1], c=kmeans.etiketler, cmap='viridis', marker='o')
plt.scatter(kmeans.merkezler[:, 0], kmeans.merkezler[:, 1], s=300, c='red', marker='X')
plt.title("K-means Sonuçları")

# SNN tahminleri
plt.subplot(1, 3, 3)
plt.scatter(normalizasyonlu_veri[:, 0], normalizasyonlu_veri[:, 1], c=etiketler, cmap='viridis', marker='o')
plt.scatter(tahmin_nokta[0], tahmin_nokta[1], c='red', s=300, marker='X')
plt.title(f"SNN Tahmini: Etiket {snn_etiket}")

plt.show()
