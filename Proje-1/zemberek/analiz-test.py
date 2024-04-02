import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from zemberek import TurkishTokenizer, morphology

# Zemberek kütüphanesi


# Köşe yazarlarının köşe yazılarının bulunduğu dizin
base_dir = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek"

# Her yazar için 20 köşe yazısı alacak şekilde eğitim verisi oluştur
training_data = defaultdict(list)
tokenizer = TurkishTokenizer()
morphology = morphology()

for author_folder in os.listdir(base_dir):
    author_path = os.path.join(base_dir, author_folder)
    if os.path.isdir(author_path):
        author = author_folder
        for file_name in os.listdir(author_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(author_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    # Zemberek ile metni işleme
                    tokens = tokenizer.tokenize(text)
                    lemmas = [morphology.lemmatize(token) for token in tokens]
                    text = " ".join(lemmas)
                    training_data[author].append(text)

# Test verisini yükleme
test_article_path = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\TEST.txt"
with open(test_article_path, "r", encoding="utf-8") as test_file:
    test_article = test_file.read()
    # Zemberek ile metni işleme
    tokens = tokenizer.tokenize(test_article)
    lemmas = [morphology.lemmatize(token) for token in tokens]
    test_article = " ".join(lemmas)

# Veri setini ayırma (Train ve Test)
X_train, X_test, y_train, y_test = train_test_split(training_data.values(), 
                                                    list(training_data.keys()), 
                                                    test_size=0.2, 
                                                    random_state=42)  # Belirli bir rastgele sayı üreteci kullanarak bölme

# TF-IDF vektörleştirici ve Naive Bayes sınıflandırıcı modeli oluşturma
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Modeli eğitme
model.fit(X_train, y_train)

# Test verisini kullanarak tahmin yapma
predicted_author = model.predict([test_article])[0]
print("Tahmin Edilen Yazar:", predicted_author)

# Değerlendirme metriklerini hesaplama
y_pred = model.predict(X_test)
authors = list(set(y_train))  # Benzersiz yazarları topla

confusion_matrix_data = confusion_matrix(y_test, y_pred, labels=authors)
precision_scores = precision_score(y_test, y_pred, average=None, labels=authors, zero_division=1)
recall_scores = recall_score(y_test, y_pred, average=None, labels=authors, zero_division=1)
f1_scores = f1_score(y_test, y_pred, average=None, labels=authors, zero_division=1)

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
