import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from zemberek import TurkishSentenceExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Zemberek kütüphanesini kullanarak Türkçe metin işleme özelliklerini tanımla
extractor = TurkishSentenceExtractor()

# Köşe yazarlarının köşe yazılarının bulunduğu dosyaların konumu
base_dir = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek"
author_dirs = ["yazar1", "yazar2", "yazar3", "yazar4", "yazar5"]

# Her yazar için minimum 20 köşe yazısı alacak şekilde eğitim verisi oluştur
training_data = defaultdict(list)
min_document_count = 20  # Her yazar için minimum 20 dosya (köşe yazısı) alacağız

for author_dir in author_dirs:
    author = re.match(r"yazar(\d+)", author_dir).group(1)  # Yazarı belirle
    author_path = os.path.join(base_dir, author_dir)
    doc_files = [f for f in os.listdir(author_path) if f.endswith(".txt")]
    if len(doc_files) < min_document_count:
        print(f"{author_dir} için yeterli sayıda dosya yok.")
        continue
    for file_name in doc_files[:min_document_count]:  # En fazla ilk 20 dosyayı al
        file_path = os.path.join(author_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            sentences = extractor.from_paragraph(text)
            random.shuffle(sentences)  # Yazıları karıştır
            training_data[author].extend(sentences)

# Oluşturulan eğitim verisini göster
for author, sentences in training_data.items():
    print(f"Yazar: {author}, Toplam Cümle Sayısı: {len(sentences)}")

# Veri setini özellik matrisine ve hedef etiketlerine dönüştürme
X = []
y = []
for author, sentences in training_data.items():
    X.extend(sentences)
    y.extend([author] * len(sentences))

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes sınıflandırıcı modeli oluşturma
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Modeli eğitme
model.fit(X_train, y_train)

# Modelin performansını değerlendirme
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix hesaplama
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=training_data.keys(), yticklabels=training_data.keys())
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()

# Modeli kullanarak yeni bir köşe yazısının yazarını tahmin etme
new_article_path = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Proje-1/zemberek/doc6.txt"
with open(new_article_path, "r", encoding="utf-8") as file:
    new_article = file.read()

predicted_author = model.predict([new_article])[0]
print("Tahmin Edilen Yazar:", predicted_author)

# Modelin çapraz doğrulama skorlarını hesaplama
cv_scores = cross_val_score(model, X, y, cv=5)  # 5 kat çapraz doğrulama
print("Çapraz Doğrulama Skorları:", cv_scores)
print("Ortalama Çapraz Doğrulama Skoru:", cv_scores.mean())
