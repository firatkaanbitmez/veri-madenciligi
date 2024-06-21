import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Parametreler
def parametreleri_ayarla():
    parametreler = {
        "random_seed": 1,  # Rastgele sayı üretici için başlangıç değeri
        "kume1_ort": [5, 5],  # 1. küme için ortalama değerler
        "kume2_ort": [0, 0],  # 2. küme için ortalama değerler
        "kume_örnek_sayisi": 200,  # Her küme için örnek sayısı
        "snn_k": 3,  # SNN algoritması için komşu sayısı
        "kmeans_k": 2,  # K-means algoritması için küme sayısı
        "kmeans_max_iterasyon": 100,  # K-means algoritması için maksimum iterasyon sayısı
        "tahmin_nokta": [0, 0]  # SNN algoritması için tahmin edilecek nokta
    }
    return parametreler

parametreler = parametreleri_ayarla()

# Veri seti oluşturma
np.random.seed(parametreler["random_seed"])  
veri_kume1 = np.random.randn(parametreler["kume_örnek_sayisi"], 2) + np.array(parametreler["kume1_ort"])  
veri_kume2 = np.random.randn(parametreler["kume_örnek_sayisi"], 2) + np.array(parametreler["kume2_ort"])  
veri = np.vstack((veri_kume1, veri_kume2))  # iki kümeyi bir araya getirir


# Veri normalizasyonu
def normalizasyon(veri):
    """
    Veriyi normalize eder. Her bir özellik (sütun) için ortalamayı çıkarır ve standart sapmaya böler.
    """
    ortalama = np.mean(veri, axis=0)  # Her sütunun ortalamasını hesaplar
    std_sapma = np.std(veri, axis=0)  # Her sütunun standart sapmasını hesaplar
    return (veri - ortalama) / std_sapma  # Veriyi normalize eder

normalizasyonlu_veri = normalizasyon(veri)

# SNN (Simple Nearest Neighbor) algoritması
class SNN:
    def __init__(self, k=3):
        self.k = k  # Kaç komşuya bakılacağını belirler

    def fit(self, veri, etiketler):
        """
        Veriyi ve etiketleri alarak modelin eğitimini yapar.
        """
        self.veri = veri
        self.etiketler = etiketler
        self.mesafe_matrisi = distance_matrix(veri, veri)  # Tüm veri noktaları arasındaki mesafeleri hesaplar
        self.benzerlik_matrisi = np.zeros(self.mesafe_matrisi.shape)  # Benzerlik matrisini oluşturur

        for i in range(len(veri)):
            komsular = np.argsort(self.mesafe_matrisi[i])[:self.k+1]  # En yakın komşuları bulur
            for j in komsular:
                self.benzerlik_matrisi[i, j] = 1  # Komşular arasında bir bağ oluşturur

    def tahmin(self, nokta):
        """
        Yeni bir noktanın etiketini tahmin eder.
        """
        mesafe = np.linalg.norm(self.veri - nokta, axis=1)  # Yeni noktanın tüm veri noktalarına olan mesafesini hesaplar
        komsular = np.argsort(mesafe)[:self.k]  # En yakın komşuları bulur
        komsu_etiketler = self.etiketler[komsular]  # Komşuların etiketlerini alır
        etiket_sayilari = np.bincount(komsu_etiketler.astype(int))  # Etiketlerin sayısını hesaplar
        return np.argmax(etiket_sayilari), etiket_sayilari  # En sık görülen etiketi döndürür

# K-means algoritması
class KMeans:
    def __init__(self, k=2, max_iterasyon=100):
        self.k = k  # Kaç küme oluşturulacağını belirler
        self.max_iterasyon = max_iterasyon  # Maksimum iterasyon sayısını belirler

    def fit(self, veri):
        """
        Veriyi alarak k-means algoritmasını uygular.
        """
        self.veri = veri
        self.merkezler = veri[np.random.choice(len(veri), self.k, replace=False)]  # Rastgele merkezler seçer
        
        for _ in range(self.max_iterasyon):
            self.etiketler = self.kumeleri_ata()  # Veriyi kümelere atar
            yeni_merkezler = self.merkezleri_hesapla()  # Yeni merkezleri hesaplar
            
            if np.all(self.merkezler == yeni_merkezler):  # Eğer merkezler değişmiyorsa döngüyü kırar
                break
            
            self.merkezler = yeni_merkezler  # Merkezleri günceller

    def kumeleri_ata(self):
        """
        Her veri noktasını en yakın merkeze atar.
        """
        mesafeler = np.array([np.linalg.norm(self.veri - merkez, axis=1) for merkez in self.merkezler])  # Her noktanın merkezlere olan mesafesini hesaplar
        return np.argmin(mesafeler, axis=0)  # Her nokta için en yakın merkezi bulur

    def merkezleri_hesapla(self):
        """
        Her kümenin yeni merkezini hesaplar.
        """
        return np.array([self.veri[self.etiketler == i].mean(axis=0) for i in range(self.k)])  # Her kümenin ortalamasını alarak yeni merkezleri belirler

# Etiketleri oluşturma
etiketler1 = np.zeros(parametreler["kume_örnek_sayisi"])  # 1. küme etiketleri
etiketler2 = np.ones(parametreler["kume_örnek_sayisi"])   # 2. küme etiketleri
etiketler = np.hstack((etiketler1, etiketler2))  # Etiketleri birleştirir

# SNN ve K-means algoritmalarını uygulama
snn = SNN(k=parametreler["snn_k"])
snn.fit(normalizasyonlu_veri, etiketler)

kmeans = KMeans(k=parametreler["kmeans_k"], max_iterasyon=parametreler["kmeans_max_iterasyon"])
kmeans.fit(normalizasyonlu_veri)


# SNN sonuçları için örnek tahmin
tahmin_nokta = np.array(parametreler["tahmin_nokta"])
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
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")

# K-means sonuçları
plt.subplot(1, 3, 2)
plt.scatter(normalizasyonlu_veri[:, 0], normalizasyonlu_veri[:, 1], c=kmeans.etiketler, cmap='viridis', marker='o')
plt.scatter(kmeans.merkezler[:, 0], kmeans.merkezler[:, 1], s=300, c='red', marker='X')
plt.title("K-means Sonuçları")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")

# SNN tahminleri
plt.subplot(1, 3, 3)
plt.scatter(normalizasyonlu_veri[:, 0], normalizasyonlu_veri[:, 1], c=etiketler, cmap='viridis', marker='o')
plt.scatter(tahmin_nokta[0], tahmin_nokta[1], c='red', s=300, marker='X')
plt.title(f"SNN Tahmini: Etiket {snn_etiket}")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")

plt.show()
