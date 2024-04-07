from collections import defaultdict
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import re
from snowballstemmer import TurkishStemmer
import seaborn as sns

# Veri klasörlerinin yolu
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/data"

# Test dosyalarının yolu
test_data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/testyazar"

# Stop words listesinin yolunu belirtin
stopwords_path = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/stopword.txt"

# Stop words listesini yükle
with open(stopwords_path, "r", encoding="utf-8") as stopwords_file:
    stop_words = stopwords_file.read().splitlines()

# Eğitim verisi
training_data = defaultdict(list)

# Köşe yazılarını yükle ve temizle
for author_folder in os.listdir(data_folder):
    author_path = os.path.join(data_folder, author_folder)
    if os.path.isdir(author_path):
        author_name = author_folder
        for file_name in os.listdir(author_path):
            file_path = os.path.join(author_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                # Veri temizleme
                cleaned_text = re.sub(r'\W', ' ', text)  # Noktalama işaretlerini temizle
                cleaned_text = re.sub(r'\d+', ' ', cleaned_text)  # Rakamları temizle
                cleaned_text = cleaned_text.lower()  # Küçük harfe dönüştür
                # İleri düzey ön işleme adımları
                # Tokenizasyon ve kök analizi gibi işlemler burada yapılabilir
                # Özellik Mühendisliği
                # Metin özelliklerini artırarak modelin performansını artırın
                # Örneğin, metnin uzunluğu, cümle sayısı, özel karakterlerin sayısı gibi özellikler eklenebilir
                
                # Stop words'leri kaldırma
                cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])
                
                # Kök analizi
                stemmer = TurkishStemmer()
                cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()])
                
                training_data[author_name].append(cleaned_text)

# Eğitim ve test verilerini ayır
X = []
y = []
for author, texts in training_data.items():
    X.extend(texts)
    y.extend([author] * len(texts))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vektörleme
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Modeli eğit
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_vectorized)

# Confusion matrix oluştur
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion matrix'i düzelt
sorted_classes = sorted(model.classes_)
correct_conf_matrix = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=np.int32)
for i, row in enumerate(conf_matrix):
    correct_conf_matrix[i] = row[np.argsort(sorted_classes)]

# Confusion matrix'i görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(correct_conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=sorted_classes, yticklabels=sorted_classes)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()

# Confusion matrix'i terminalden de çıktı olarak al
print("Confusion Matrix:")
print(correct_conf_matrix)

# Test verileri için tahmin yap
for file_name in os.listdir(test_data_folder):
    file_path = os.path.join(test_data_folder, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        # Veri temizleme
        cleaned_text = re.sub(r'\W', ' ', text)  
        cleaned_text = re.sub(r'\d+', ' ', cleaned_text)  
        cleaned_text = cleaned_text.lower()  
        # Stop words'leri kaldırma
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])
        # Kök analizi
        stemmer = TurkishStemmer()
        cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()])
        # Tahmin yap
        prediction = model.predict(vectorizer.transform([cleaned_text]))[0]
        print(f"{file_name}: Gerçek Yazar: {file_name.split('.')[0]}, Tahmin Edilen Yazar: {prediction}")
