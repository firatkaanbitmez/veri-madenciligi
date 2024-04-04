import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from zemberek import Zemberek

# Veri kümesinin ve test verisinin dizinleri
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Proje-1/zemberek"
test_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Proje-1/zemberek/testyazar"

# Zemberek nesnesini oluştur
zemberek = Zemberek()

# Veri ön işleme ve vektörleştirme adımları
veri = []
etiketler = []

for yazar_index, yazar_klasoru in enumerate(os.listdir(data_folder), start=1):
    for dosya in os.listdir(os.path.join(data_folder, yazar_klasoru)):
        dosya_yolu = os.path.join(data_folder, yazar_klasoru, dosya)
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            metin = f.read()
        metin = re.sub(r'\d+', '', metin)  
        metin = re.sub(r'[^\w\s]', '', metin)  
        metin = metin.lower()
        kelimeler = zemberek.kelime_cozumle(metin)
        metin = ' '.join([kelime.kok().icerik() for kelime in kelimeler])
        veri.append(metin)
        etiketler.append(yazar_index)

X_egitim, X_test, y_egitim, y_test = train_test_split(veri, etiketler, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_egitim_vetor = vectorizer.fit_transform(X_egitim)

model = MultinomialNB()
model.fit(X_egitim_vetor, y_egitim)

# Test verisinin değerlendirilmesi
X_test = []
y_test = []

for dosya in os.listdir(test_folder):
    dosya_yolu = os.path.join(test_folder, dosya)
    with open(dosya_yolu, 'r', encoding='utf-8') as f:
        metin = f.read()
    X_test.append(metin)
    y_test.append(int(dosya[4]))

X_test_vector = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vector)

# Sonuçların yazdırılması
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("Karmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
