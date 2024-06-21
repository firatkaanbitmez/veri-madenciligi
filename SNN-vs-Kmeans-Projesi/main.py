import numpy as np
import matplotlib.pyplot as plt

# Parametreler fonksiyonu
def Parametreler():
    parametreler = {
        "rastgele_seed": 42,  # Rastgelelik için seed değeri
        "kume1_ort": [1, 1],  # 1. kümenin ortalama değeri
        "kume2_ort": [5, 4],  # 2. kümenin ortalama değeri
        "kume_std": 1.0,  # Kümelerin standart sapması
        "her_kume_icin_ornek_sayisi": 100,  # Her küme için örnek sayısı
        "snn_eps": 0.7,  # SNN algoritması için komşuluk eşiği
        "snn_min_pts": 5,  # SNN algoritması için minimum komşu sayısı
        "kmeans_k": 2,  # K-means algoritması için küme sayısı
        "kmeans_max_iter": 100  # K-means algoritması için maksimum iterasyon sayısı
    }
    return parametreler

# Parametreleri yükle
parametreler = Parametreler()

# Veri seti oluşturma
np.random.seed(parametreler["rastgele_seed"])

# İki küme oluşturma
veri_kume1 = np.random.randn(parametreler["her_kume_icin_ornek_sayisi"], 2) * parametreler["kume_std"] + parametreler["kume1_ort"]
veri_kume2 = np.random.randn(parametreler["her_kume_icin_ornek_sayisi"], 2) * parametreler["kume_std"] + parametreler["kume2_ort"]

veri = np.vstack((veri_kume1, veri_kume2))
etiketler = np.hstack((np.zeros(parametreler["her_kume_icin_ornek_sayisi"]), np.ones(parametreler["her_kume_icin_ornek_sayisi"])))  # Küme etiketleri

# Veri setini görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(veri[:, 0], veri[:, 1], c=etiketler, cmap='viridis', marker='o', edgecolors='k')
plt.title('Oluşturulan İki Boyutlu Veri Seti')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.colorbar()
plt.grid(True)
plt.show()

# Ortak En Yakın Komşu (Shared Nearest Neighbor, SNN) Algoritması
class SNN:
    def __init__(self, eps=1.0, min_pts=5):
        self.eps = eps  # Komşuluk eşiği
        self.min_pts = min_pts  # Minimum komşu sayısı

    def fit(self, veri):
        self.veri = veri
        self.n = len(veri)
        self.etiketler = np.full(self.n, -1)  # -1: Gürültü, 0, 1, 2, ...: Küme etiketleri
        self.kume_sayisi = 0

        for i in range(self.n):
            if self.etiketler[i] == -1:  # Daha önce etiketlenmemişse
                self.kume_genislet(i)

    def kume_genislet(self, nokta_index):
        komsular = self.ortak_komsulari_bul(nokta_index)
        
        # Noktanın bir küme oluşturmak için yeterli komşusu olup olmadığını kontrol et
        if len(komsular) < self.min_pts:
            self.etiketler[nokta_index] = -1  # Gürültü
            return False
        else:
            self.kume_sayisi += 1
            self.etiketler[nokta_index] = self.kume_sayisi
            
            # Komşuların küme etiketlerini atama
            for komsu in komsular:
                if self.etiketler[komsu] == -1:  # Eğer komşu gürültü ise
                    self.etiketler[komsu] = self.kume_sayisi
                elif self.etiketler[komsu] == 0:  # Eğer komşu henüz atanmadıysa
                    self.etiketler[komsu] = self.kume_sayisi
                    self.kume_genislet(komsu)  # Küme genişletme işlemini tekrarla
            return True

    def ortak_komsulari_bul(self, nokta_index):
        komsular = []
        for i in range(self.n):
            if i != nokta_index:
                mesafe = np.linalg.norm(self.veri[nokta_index] - self.veri[i])
                if mesafe < self.eps:
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
        
        for iterasyon in range(self.max_iter):
            # Veri noktalarını en yakın merkeze ata
            self.etiketler = self.kumelere_ata()
            
            # Yeni merkezleri güncelle
            yeni_merkezler = self.merkezleri_guncelle()
            
            # Merkezlerin değişip değişmediğini kontrol et
            if np.allclose(self.merkezler, yeni_merkezler):
                print(f"Iterasyon {iterasyon}: Kümeler sabitlendi.")
                break
            self.merkezler = yeni_merkezler
            print(f"Iterasyon {iterasyon}: Merkezler güncellendi.")
        
        print("Son Merkezler: ", self.merkezler)

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
snn = SNN(eps=parametreler["snn_eps"], min_pts=parametreler["snn_min_pts"])
snn.fit(veri)
snn_etiketler = snn.etiketler

kmeans = KMeans(k=parametreler["kmeans_k"], max_iter=parametreler["kmeans_max_iter"])
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

# SNN ve K-means sonuçlarını terminalde yazdırma
print("SNN Kümeleme Sonuçları:")
print(snn_etiketler)

print("\nK-means Kümeleme Sonuçları:")
print(kmeans_etiketler)
