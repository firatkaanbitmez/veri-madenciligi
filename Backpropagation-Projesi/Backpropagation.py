import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# ÖNEMLİ 
# Debug modu: True ise işlemler çıktı ekranına yazdırılır, False ise yazdırılmaz ***
DEBUG_MODE = True

# Sigmoid aktivasyon fonksiyonu
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid fonksiyonunun türevi
def sigmoid_turevi(x):
    return x * (1 - x)

# Model parametrelerinin başlatılması
def parametreleri_baslat(giris_boyutu, gizli_boyut, cikis_boyutu):
    W1 = np.random.randn(giris_boyutu, gizli_boyut)  # Gizli katman ağırlıklar
    b1 = np.zeros((1, gizli_boyut))     # Gizli katman bias değerleri
    W2 = np.random.randn(gizli_boyut, cikis_boyutu)     # Gizli katman ağırlık
    b2 = np.zeros((1, cikis_boyutu))        # Çıkış bias
    return W1, b1, W2, b2

# İleri yayılım işlemi
def ileri_yayilim(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1     # Giriş verisini gizli katmana aktarma
    A1 = sigmoid(Z1)        # Gizli katman çıkışlarını sigmoid fonksiyonundan geçirme
    Z2 = np.dot(A1, W2) + b2        # Gizli katman çıkışlarını çıkış katmanına aktarma
    A2 = sigmoid(Z2)        # Çıkış katman çıkışlarını sigmoid fonksiyonundan geçirme
    return Z1, A1, Z2, A2

# Hata hesaplama
def hata_hesapla(A2, Y):
    m = Y.shape[0]          # Veri noktalarının sayısını alma
    hata = -np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))) / m
    return hata

# Geri yayılım işlemi
def geri_yayilim(X, Y, Z1, A1, Z2, A2, W1, W2, b1, b2, ogrenme_orani):
    m = X.shape[0]      # Veri noktalarının sayısını alma
    dZ2 = A2 - Y        # çıkışta hata hesaplama
    
    # Çıkış katmanındaki ağırlıkların güncellenmesi
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    
    # Gizli katmandaki hata hesaplama
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_turevi(A1)
    
    # Gizli katmandaki ağırlıkların güncellenmesi
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    
    # Ağırlıkların güncellenmesi
    W1 -= ogrenme_orani * dW1
    b1 -= ogrenme_orani * db1
    W2 -= ogrenme_orani * dW2
    b2 -= ogrenme_orani * db2

    if DEBUG_MODE:
        # geri yayılım hesaplamalarını çıktı oalrak yazdırma
        print("Geri Yayılım Hesaplamaları:")
        print("dW1:", dW1)
        print("db1:", db1)
        print("dW2:", dW2)
        print("db2:", db2)
    
    return W1, b1, W2, b2

# Sinir ağı modelinin eğitimi
def sinir_agi_egitimi(X_train, y_train, X_test, y_test, giris_boyutu, gizli_boyut, cikis_boyutu, ogrenme_orani, iterasyon_sayisi):
    # Parametreleri başlatma
    W1, b1, W2, b2 = parametreleri_baslat(giris_boyutu, gizli_boyut, cikis_boyutu)
    y_train = y_train.astype(int)  # y_train'i tamsayıya dönüştür
    y_test = y_test.astype(int)    # y_test'i tamsayıya dönüştür
    
    # Eğitim ve test verisi için maliyet ve doğruluk metriklerinin listeleri
    egitim_maliyetleri = []
    test_maliyetleri = []
    egitim_dogruluklari = []
    test_dogruluklari = []
    
    # Eğitim döngüsü
    for iterasyon in range(iterasyon_sayisi):
        # İleri ve geri yayılım
        Z1, A1, Z2, A2 = ileri_yayilim(X_train, W1, b1, W2, b2)
        W1, b1, W2, b2 = geri_yayilim(X_train, y_train.reshape(-1, 1), Z1, A1, Z2, A2, W1, W2, b1, b2, ogrenme_orani)
        
        # Eğitim verisi üzerinde maliyeti hesapla ve kaydet
        egitim_maliyeti = hata_hesapla(A2, y_train.reshape(-1, 1))
        egitim_maliyetleri.append(egitim_maliyeti)
        
        # Eğitim verisi üzerinde doğruluk hesapla ve kaydet
        egitim_tahminleri = tahmin_et(X_train, W1, b1, W2, b2)
        egitim_tahminleri = (egitim_tahminleri > 0.5).astype(int)
        egitim_dogrulugu = accuracy_score(y_train, egitim_tahminleri)
        egitim_dogruluklari.append(egitim_dogrulugu)
        
        # Test verisi üzerinde maliyeti hesapla ve kaydet
        Z1_test, A1_test, Z2_test, A2_test = ileri_yayilim(X_test, W1, b1, W2, b2)
        test_maliyeti = hata_hesapla(A2_test, y_test.reshape(-1, 1))
        test_maliyetleri.append(test_maliyeti)
        
        # Test verisi üzerinde doğruluk hesapla ve kaydet
        test_tahminleri = tahmin_et(X_test, W1, b1, W2, b2)
        test_dogrulugu = accuracy_score(y_test, test_tahminleri)
        test_dogruluklari.append(test_dogrulugu)
        
        # Her 100 iterasyonda bir eğitim ve test verisi için maliyeti ve doğruluğu yazdır
        if iterasyon % 100 == 0:
            if DEBUG_MODE:
                print(f"Iterasyon {iterasyon}:")
                print(f"  Eğitim Maliyeti: {egitim_maliyeti}, Eğitim Doğruluğu: {egitim_dogrulugu}")
                print(f"  Test Maliyeti: {test_maliyeti}, Test Doğruluğu: {test_dogrulugu}")
    
    return W1, b1, W2, b2, egitim_maliyetleri, test_maliyetleri, egitim_dogruluklari, test_dogruluklari

# Modelin test verisi üzerinde tahmin yapması
def tahmin_et(X_test, W1, b1, W2, b2):
    _, _, _, tahminler = ileri_yayilim(X_test, W1, b1, W2, b2)
    tahminler = (tahminler > 0.5).astype(int)
    return tahminler

# Veriyi yükleme ve eğitim/test kümelerine ayırma
veri_yolu = "C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Backpropagation-Projesi\\data.txt"  # Veri yolu belirtilmeli
veri = np.loadtxt(veri_yolu)
X = veri[:, :-1]
y = veri[:, -1]
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparametreler
giris_boyutu = X_egitim.shape[1]
gizli_boyut = 10  # Değiştirilebilir
cikis_boyutu = 1   # İkili sınıflandırma varsayımı
ogrenme_orani = 0.01
iterasyon_sayisi = 1000

# Modeli eğitme
W1, b1, W2, b2, egitim_maliyetleri, test_maliyetleri, egitim_dogruluklari, test_dogruluklari = sinir_agi_egitimi(X_egitim, y_egitim, X_test, y_test, giris_boyutu, gizli_boyut, cikis_boyutu, ogrenme_orani, iterasyon_sayisi)

# Sonuçları görselleştirme
plt.figure(figsize=(12, 5))

# Eğitim ve test maliyetlerini çizdirme
plt.subplot(1, 2, 1)
plt.plot(range(iterasyon_sayisi), egitim_maliyetleri, label='Eğitim')
plt.plot(range(iterasyon_sayisi), test_maliyetleri, label='Test')
plt.xlabel('Iterasyon')
plt.ylabel('Maliyet')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()

# Eğitim ve test doğruluklarını çizdirme
plt.subplot(1, 2, 2)
plt.plot(range(iterasyon_sayisi), egitim_dogruluklari, label='Eğitim')
plt.plot(range(iterasyon_sayisi), test_dogruluklari, label='Test')
plt.xlabel('Iterasyon')
plt.ylabel('Doğruluk')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()

plt.tight_layout()
plt.show()
