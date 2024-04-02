import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Köşe yazarlarının köşe yazılarının bulunduğu dizin
base_dir = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek"

# Her yazar için 20 köşe yazısı alacak şekilde eğitim verisi oluştur
training_data = defaultdict(list)

for author_folder in os.listdir(base_dir):
    author_path = os.path.join(base_dir, author_folder)
    if os.path.isdir(author_path):
        author = author_folder
        for file_name in os.listdir(author_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(author_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    training_data[author].append(text)

# Test verisini yükleme
test_article_path = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\TEST.txt"
with open(test_article_path, "r", encoding="utf-8") as test_file:
    test_article = test_file.read()

# Veri setini özellik matrisine ve hedef etiketlerine dönüştürme
X_train = []
y_train = []
for author, articles in training_data.items():
    X_train.extend(articles)
    y_train.extend([author] * len(articles))

# TF-IDF vektörleştirici ve Naive Bayes sınıflandırıcı modeli oluşturma
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Modeli eğitme
model.fit(X_train, y_train)

# Test verisini kullanarak tahmin yapma
predicted_author = model.predict([test_article])[0]
print("Tahmin Edilen Yazar:", predicted_author)

# Confusion Matrix hesaplama
y_pred = model.predict(X_train)
authors = list(training_data.keys())
confusion_matrix_data = confusion_matrix(y_train, y_pred, labels=authors)

# Hassasiyet, Duyarlılık ve F1 Skoru hesaplama
precision_scores = precision_score(y_train, y_pred, average=None, labels=authors, zero_division=1)
recall_scores = recall_score(y_train, y_pred, average=None, labels=authors, zero_division=1)
f1_scores = f1_score(y_train, y_pred, average=None, labels=authors, zero_division=1)

# Confusion Matrix görselleştirme
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix_data, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xticks(ticks=range(len(authors)), labels=authors, rotation=45)
plt.yticks(ticks=range(len(authors)), labels=authors)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Confusion Matrix')

for i in range(len(authors)):
    for j in range(len(authors)):
        plt.text(j, i, confusion_matrix_data[i, j], ha='center', va='center', color='red' if i != j else 'black')

plt.tight_layout()
plt.show()

# Raporlama
print("\n--- RAPOR ---")
print("Yazar\t\tHassasiyet\tDuyarlılık\tF1 Skoru")
for i, author in enumerate(authors):
    print(f"{author}\t\t{precision_scores[i]:.2f}\t\t{recall_scores[i]:.2f}\t\t{f1_scores[i]:.2f}")
