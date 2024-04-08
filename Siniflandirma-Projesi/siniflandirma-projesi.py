from collections import defaultdict
import os
import re
import numpy as np
from snowballstemmer import TurkishStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# Veri klasörü, test verisi klasörü ve stop word dosyası yollarını belirtin
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/data"
test_data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/testyazar"
stopwords_path = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/stopword.txt"

# Stop word dosyasını okuyarak, metinlerdeki gereksiz kelimeleri filtrelemek için bir stop word listesi
with open(stopwords_path, "r", encoding="utf-8") as stopwords_file:
    stop_words = stopwords_file.read().splitlines()

# Eğitim verisi için bir defaultdict oluşturuyoruz. Bu, her yazarın makalelerinin bir listesini saklamak için kullanılacaktır.
training_data = defaultdict(list)

# Eğitim verisini okuyun ve temizleyin
for author_folder in os.listdir(data_folder):
    author_path = os.path.join(data_folder, author_folder)
    if os.path.isdir(author_path):
        author_name = author_folder
        for file_name in os.listdir(author_path):
            file_path = os.path.join(author_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                # Metin temizleme işlemleri
                cleaned_text = re.sub(r'\W', ' ', text)   #Noktalama işaretlerinin kaldırılması
                cleaned_text = re.sub(r'\d+', ' ', cleaned_text) #Sayıların kaldırılması
                cleaned_text = cleaned_text.lower()   #Metnin küçük harflere dönüştürülmesi
                cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words]) #Stop words'lerin kaldırılması
                # Türkçe kök bulma işlemi
                stemmer = TurkishStemmer()
                cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()])
                # Eğitim verisine temizlenmiş metni ekleyin
                training_data[author_name].append(cleaned_text)

# Eğitim ve test verilerini hazırlanması
#Eğitim verileri (X) ve etiketleri (y) oluşturuluyor. Burada, metinler X listesine eklenirken, her metnin etiketi y listesine ekleniyor.
X = []
y = []
for author, texts in training_data.items():
    X.extend(texts)
    y.extend([author] * len(texts))

# Model oluşturma
#Pipeline, ardışık işlemleri sıralı olarak gerçekleştirmenizi sağlar. Bu durumda, TF-IDF vektörleme işlemi ve SVC (Support Vector Classifier) sınıflandırma algoritması sırayla uygulanır.
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC())  
])

# GridSearchCV ile en iyi parametreleri bulma
#GridSearchCV, bir model için en iyi hiperparametreleri bulmanızı sağlayan bir cross-validation aracıdır. Bu durumda, SVC için C, gamma ve kernel parametreleri için farklı değerler denenecek ve en iyi parametreler bulunacaktır.
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("GridSearchCV ile en iyi parametrelerin bulunması:")
print("------------------------------------------------")
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)
print("\n")

best_model = grid_search.best_estimator_
#En iyi model seçilir ve eğitim verisi ile eğitilir. Ayrıca, doğrulama verisi için train_test_split kullanılarak veri seti bölünür.
# Eğitim ve doğrulama verisi için modeli eğitin
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)


#Modelin performansını değerlendirmek için confusion matrix hesaplanır.
y_pred = best_model.predict(X_val)
conf_matrix = confusion_matrix(y_val, y_pred)

#Confusion matrix, görselleştirilerek daha anlaşılır hale getirilir.
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')
plt.show()

# Sınıflandırma raporu
print("Sınıflandırma Raporu:")
print("--------------------")
print(classification_report(y_val, y_pred))
print("\n")

# Test verisi üzerinde tahminler
print("Tahminler:")
print("----------------------------")
predictions = []
for i, file_name in enumerate(sorted(os.listdir(test_data_folder), key=lambda x: int(re.findall(r'\d+', x)[0]))):
    file_path = os.path.join(test_data_folder, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        # Test metnini temizleme ve kök bulma işlemleri
        cleaned_text = re.sub(r'\W', ' ', text)  
        cleaned_text = re.sub(r'\d+', ' ', cleaned_text)  
        cleaned_text = cleaned_text.lower()  
        cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])
        stemmer = TurkishStemmer()
        cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()])
        # Model ile tahmin yapma
        prediction = best_model.predict([cleaned_text])[0]
        predictions.append((file_name, prediction))

# Tahminleri detaylı olarak yazdırma
for file_name, prediction in predictions:
    print(f"{file_name}: Tahmin Edilen Yazar: {prediction}")

