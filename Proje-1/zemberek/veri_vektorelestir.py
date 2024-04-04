from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

# Veri kümesinin dizini
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Proje-1/zemberek"

# Tüm yazar klasörlerini al
yazar_klasorleri = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# Veri ve etiket listelerini oluştur
veri = []
etiketler = []

# Her bir yazar için
for idx, yazar_klasoru in enumerate(yazar_klasorleri, start=1):
    # Tüm dosyaları al
    dosya_listesi = os.listdir(yazar_klasoru)
    
    # Her bir dosya için
    for dosya in dosya_listesi:
        dosya_yolu = os.path.join(yazar_klasoru, dosya)
        
        # Dosyayı oku
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            metin = f.read()
        
        # Veri listesine ekle
        veri.append(metin)
        etiketler.append(idx)  # Her bir yazar için bir etiket
        
# Veriyi eğitim ve test kümelerine bölelim
X_egitim, X_test, y_egitim, y_test = train_test_split(veri, etiketler, test_size=0.2, random_state=42)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
X_egitim_vetor = vectorizer.fit_transform(X_egitim)

# Modeli oluştur ve eğit
model = MultinomialNB()
model.fit(X_egitim_vetor, y_egitim)
