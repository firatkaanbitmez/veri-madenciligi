import os
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Eğitim verilerinin bulunduğu dizin
base_dir = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek"

# Eğitim verisi için defaultdict kullanarak bir sözlük oluşturuluyor
training_data = defaultdict(list)

# Her yazar için eğitim verisini toplama
for author_folder in os.listdir(base_dir):
    author_path = os.path.join(base_dir, author_folder)
    if os.path.isdir(author_path):
        author = author_folder
        for file_name in os.listdir(author_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(author_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    training_data[str(author)].append((file_name, text))

# Test verisi yükleniyor
test_article_path = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek\testyazar\Test1.txt"
with open(test_article_path, "r", encoding="utf-8") as test_file:
    lines = test_file.readlines()
    # Test verisi işleniyor
    y_test = [line.strip().split(":")[1].strip() for line in lines if line.startswith("Yazar")]
    test_article = "".join([line.strip() for line in lines if line.startswith("Yazı")])

# Eğitim verisi hazırlanıyor
X_train = []
y_train = []
for author, articles in training_data.items():
    for article_name, article_text in articles:
        X_train.append(article_text)
        y_train.append(author)

# Türkçe stop-words listesi
turkish_stop_words = ["acaba", "ama", "aslında", ...]

# Model oluşturuluyor
model = make_pipeline(TfidfVectorizer(stop_words=turkish_stop_words), MultinomialNB())
model.fit(X_train, y_train)

# Test verisi için tahmin yapılıyor
predicted_author = model.predict([test_article])[0]
print("Tahmin Edilen Yazar:", predicted_author)

# Test verisi ile modelin performansı ölçülüyor
X_test = [test_article] 
y_test = ["yazar1"]   # Burada test verisi için gerçek etiketler verilmelidir.

y_pred = model.predict(X_test)

# Confusion Matrix ve diğer metrikler hesaplanıyor
confusion_matrix_data = confusion_matrix(y_test, y_pred, labels=model.classes_)
precision_scores = precision_score(y_test, y_pred, average=None, labels=model.classes_, zero_division=1)
recall_scores = recall_score(y_test, y_pred, average=None, labels=model.classes_, zero_division=1)
f1_scores = f1_score(y_test, y_pred, average=None, labels=model.classes_, zero_division=1)

# Confusion Matrix görselleştiriliyor
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix_data, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(ticks=range(len(model.classes_)), labels=model.classes_, rotation=45)
plt.yticks(ticks=range(len(model.classes_)), labels=model.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')

for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(j, i, confusion_matrix_data[i, j], ha='center', va='center', color='red' if i!= j else 'black')

plt.tight_layout()
plt.show()

# Performans metrikleri raporlanıyor
print("\n--- RAPOR ---")
print("Yazar\t\tHassasiyet\tDuyarlılık\tF1 Skoru")
for i, author in enumerate(model.classes_):
    print(f"{author}\t\t{precision_scores[i]:.2f}\t\t{recall_scores[i]:.2f}\t\t{f1_scores[i]:.2f}")
