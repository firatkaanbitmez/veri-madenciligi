# Giriş ve Ön İşleme
"""
Bu proje, bir yapay sinir ağı modeli kullanarak meme kanseri teşhisi koymayı amaçlamaktadır.
Veri seti, Duke Üniversitesi Meme Kanseri Veri Deposu'ndan alınmıştır.
Toplamda 700 örnek içermektedir. Her bir örnek, 10 özellikten oluşmaktadır ve son sütun,
meme kanseri teşhisini temsil eden 0 veya 1 etiketlerini içermektedir.

Ön işleme adımları şunlardır:
1. Veri seti karıştırılarak rastgele örneklerin sırası değiştirildi.
2. Özellikler (girişler) ve etiketler (çıktılar) ayrıldı.
3. Veri seti, eğitim ve test veri setleri olmak üzere %10 test verisi olacak şekilde bölündü.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Veri setini yükleme ve ön işleme
veri_yolu = "C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Backpropagation-Projesi\\data.txt"
veri = np.loadtxt(veri_yolu)
print("Veri seti boyutu:", veri.shape)

# Eksik değerleri kontrol etme
eksik_deger_sayisi = np.isnan(veri).sum()
if eksik_deger_sayisi > 0:
    print("Veri setinde eksik değerler bulunuyor.")

# Aykırı değerleri kontrol etme
Q1 = np.percentile(veri, 25, axis=0)
Q3 = np.percentile(veri, 75, axis=0)
IQR = Q3 - Q1
aykiri_deger_sayisi = ((veri < (Q1 - 1.5 * IQR)) | (veri > (Q3 + 1.5 * IQR))).sum()
if aykiri_deger_sayisi > 0:
    print("Veri setinde aykırı değerler bulunuyor.")

# Özellikler (girişler) ve etiketler (çıktılar) ayırma
etiketler = veri[:, -1]
ozellikler = veri[:, :-1]

# Normalizasyon
scaler = StandardScaler()
ozellikler_normal = scaler.fit_transform(ozellikler)

# Veri setini eğitim, doğrulama ve test veri setlerine bölmek
x_egitim, x_test, y_egitim, y_test = train_test_split(ozellikler_normal, etiketler, test_size=0.2, random_state=42)
x_egitim, x_dogrulama, y_egitim, y_dogrulama = train_test_split(x_egitim, y_egitim, test_size=0.25, random_state=42)  # %60 train, %20 validation, %20 test

print("Eğitim veri seti boyutu:", x_egitim.shape, "Doğrulama veri seti boyutu:", x_dogrulama.shape, "Test veri seti boyutu:", x_test.shape)

# Verileri karıştırma ve özellikleri ile etiketleri ayırma
np.random.shuffle(veri)
etiketler = veri[:, 0]
ozellikler = np.delete(veri, [0], axis=1)
x_egitim, x_test, y_egitim, y_test = train_test_split(ozellikler, etiketler, test_size=0.1)
print("Eğitim veri seti boyutu:", x_egitim.shape, "Test veri seti boyutu:", x_test.shape)

# Ağ parametrelerini başlatma
gizli_katman_boyutu = 72
gizli_katman = np.zeros(gizli_katman_boyutu)
agirliklar_giris_gizli = np.random.random((len(ozellikler[0]), gizli_katman_boyutu))
cikti_katman_boyutu = 2
cikti_katman = np.zeros(cikti_katman_boyutu)
gizli_agirliklar = np.random.random((gizli_katman_boyutu, cikti_katman_boyutu))

# Yapay Sinir Ağı Modeli
""" 
Yapay sinir ağı modeli, 10 özellikten oluşan bir giriş katmanı, 72 düğümlü bir gizli katman ve
iki düğümlü bir çıkış katmanından oluşmaktadır. Gizli katman boyutu, deneyimler sonucunda
seçilmiştir ve ağın karmaşıklığını artırarak daha iyi genelleme sağlaması amaçlanmıştır.
Giriş ve çıkış katmanlarındaki düğüm sayıları, veri setinin doğasından kaynaklanmaktadır.
"""

# Toplam fonksiyonu hesaplama
def toplam_fonksiyonu(agirliklar, index_kilitli_kolon, x):
    toplam = 0
    for i in range(len(x)):
        toplam += x[i] * agirliklar[i][index_kilitli_kolon]
    return toplam

# Katmanı aktifleştirme
def katmani_aktiflestir(katman, agirliklar, x):
    for i in range(len(katman)):
        katman[i] = 1.7159 * np.tanh(2.0 * toplam_fonksiyonu(agirliklar, i, x) / 3.0)


# Aktivasyon Fonksiyonları
"""
Aktivasyon fonksiyonları olarak hiperbolik tanjant (tanh) ve softmax fonksiyonları kullanılmıştır.
Hiperbolik tanjant fonksiyonu, aralığını [-1, 1] olarak sınırlayarak gradyanın hızlı bir şekilde
hesaplanmasına yardımcı olur. Softmax fonksiyonu ise çıkış katmanında çok sınıflı sınıflandırma
problemlerinde kullanılan bir fonksiyondur ve çıkışı olasılık dağılımı şeklinde normalize eder.
"""

# Softmax aktivasyon fonksiyonu
def softmax(katman):
    exp_degerler = np.exp(katman - np.max(katman))
    return exp_degerler / exp_degerler.sum()




# Parametrelerin Başlatılması ve Geri Yayılım Algoritması
"""
Ağırlıklar, rastgele değerlerle başlatılmıştır. Bu, ağın öğrenmeye rastgele bir şekilde başlamasını
ve farklı minimumlara sıkışmaktan kaçınmasını sağlar. Ayrıca, öğrenme oranı (-1) ve diğer parametreler
geri yayılım algoritmasında kullanılır.

Geri yayılım algoritması, ağın öğrenme sürecini temsil eder. Bu algoritma, ağın çıkışındaki hataları
geriye doğru yayarak her bir ağırlığın güncellenmesini sağlar.
"""

# Ağırlıkları güncelleme
def agirliklari_guncelle(ogrenme_orani, agirliklar, gradyan, aktivasyon):
    for i in range(len(agirliklar)):
        for j in range(len(agirliklar[i])):
            agirliklar[i][j] += (ogrenme_orani * gradyan[j] * aktivasyon[i])

# Geri yayılım algoritması
def geri_yayilim(gizli_katman, cikti_katman, one_hot_encoding, ogrenme_orani, x):
    cikti_turev = (1 - cikti_katman) * cikti_katman
    cikti_gradyan = cikti_turev * (one_hot_encoding - cikti_katman)
    gizli_turev = (1 - gizli_katman) * (1 + gizli_katman)
    gizli_gradyan = np.dot(cikti_gradyan, gizli_agirliklar.T) * gizli_turev
    agirliklari_guncelle(ogrenme_orani, gizli_agirliklar, cikti_gradyan, gizli_katman)
    agirliklari_guncelle(ogrenme_orani, agirliklar_giris_gizli, gizli_gradyan, x)


# Eğitim ve Test Aşamaları
"""
Eğitim aşamasında, ağa eğitim veri seti üzerinden ilerlerken geri yayılım algoritması kullanılır.
Her bir örnek için tahmin yapılır, hata hesaplanır ve ağırlıklar güncellenir. Eğitim doğruluğu
hesaplanarak modelin performansı değerlendirilir.

Test aşamasında, ağa ayrılmış olan test veri seti üzerinden ilerlenir ve modelin performansı
test edilir. Test doğruluğu hesaplanarak modelin genelleme yeteneği değerlendirilir.
"""

# One-hot encoding
one_hot_encoding = np.eye(cikti_katman_boyutu)

# Eğitim
egitim_dogru_sayisi = 0
for i in range(len(x_egitim)):
    katmani_aktiflestir(gizli_katman, agirliklar_giris_gizli, x_egitim[i])
    katmani_aktiflestir(cikti_katman, gizli_agirliklar, gizli_katman)
    cikti_katman = softmax(cikti_katman)
    egitim_dogru_sayisi += 1 if y_egitim[i] == np.argmax(cikti_katman) else 0
    geri_yayilim(gizli_katman, cikti_katman, one_hot_encoding[int(y_egitim[i])], -1, x_egitim[i])

egitim_dogruluk_orani = egitim_dogru_sayisi / len(x_egitim)
print("Eğitim doğruluğu: %s / %s (%.2f%%) " % (egitim_dogru_sayisi, len(x_egitim),  egitim_dogruluk_orani * 100))

# Test
test_dogru_sayisi = 0
for i in range(len(x_test)):
    katmani_aktiflestir(gizli_katman, agirliklar_giris_gizli, x_test[i])
    katmani_aktiflestir(cikti_katman, gizli_agirliklar, gizli_katman)
    cikti_katman = softmax(cikti_katman)
    test_dogru_sayisi += 1 if y_test[i] == np.argmax(cikti_katman) else 0

test_dogruluk_orani = test_dogru_sayisi / len(x_test)
print("Test doğruluğu: %s / %s (%.2f%%)" % (test_dogru_sayisi, len(x_test), test_dogruluk_orani * 100))

# Hiperparametre optimizasyonu için kullanılacak model
model = MLPClassifier()

# Hiperparametre aralıklarını belirleme
param_distributions = {
    'hidden_layer_sizes': [(50,), (60,), (70,), (80,), (90,), (100,)],
    'learning_rate_init': [0.01, 0.05, 0.1],
}

# Hiperparametre optimizasyonu
optimizer = RandomizedSearchCV(model, param_distributions, n_iter=10, scoring='accuracy', cv=3, random_state=42)
optimizer.fit(x_egitim, y_egitim)

# En iyi hiperparametreleri bulma
print("En iyi hiperparametreler:", optimizer.best_params_)

# En iyi modeli seçme
en_iyi_model = optimizer.best_estimator_

# Modelin performansını değerlendirme
y_pred = en_iyi_model.predict(x_test)
dogruluk = accuracy_score(y_test, y_pred)
hassasiyet = precision_score(y_test, y_pred)
ozgulluk = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test doğruluğu:", dogruluk)
print("Hassasiyet:", hassasiyet)
print("Özgüllük:", ozgulluk)    
print("F1 Skoru:", f1)