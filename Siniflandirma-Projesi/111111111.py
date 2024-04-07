from collections import defaultdict
import os
import re
import numpy as np
from snowballstemmer import TurkishStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# Veri klasörlerinin yolu
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/data"

# Test dosyalarının yolu
test_data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/testyazar"

# Stop words listesinin yolu
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

# Model oluşturma
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC())  # SVM kullanımı
])

# Model için parametre aralığını belirle
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

# GridSearchCV ile en iyi parametreleri bul
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)

# Modeli en iyi parametrelerle tekrar eğit
best_model = grid_search.best_estimator_

# Modeli eğitim ve doğrulama verileriyle eğitin
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)

# Test verileri üzerinde tahmin yap
y_pred = best_model.predict(X_val)

# Confusion matrix oluştur
conf_matrix = confusion_matrix(y_val, y_pred)

# Confusion matrix'i görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma raporu
print("Sınıflandırma Raporu:")
print(classification_report(y_val, y_pred))

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
        prediction = best_model.predict([cleaned_text])[0]
        print(f"{file_name}: Gerçek Yazar: {file_name.split('.')[0]}, Tahmin Edilen Yazar: {prediction}")
