import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Veri seti dizini
data_dir = "C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Sınıflandırma-Projesi\\Veri-Test-kümesi"

# Verileri ve etiketleri depolamak için boş listeler oluştur
documents = []
labels = []

# Her yazar için veri setini oku ve işle
for author_folder in os.listdir(data_dir):
    author_path = os.path.join(data_dir, author_folder)
    if os.path.isdir(author_path):
        for file_name in os.listdir(author_path):
            file_path = os.path.join(author_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    documents.append(text)
                    labels.append(author_folder)

# Kelime sayılarını hesaplamak için bir vektörleyici oluştur
vectorizer = CountVectorizer()

# Metin belgelerini vektörize et
X = vectorizer.fit_transform(documents)

# Etiketleri sayısal olarak kodla
label_counts = Counter(labels)
label_map = {label: i for i, label in enumerate(label_counts.keys())}
y = [label_map[label] for label in labels]

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = MultinomialNB()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Doğruluk değerini hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

# Karışıklık matrisini oluştur
conf_matrix = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", conf_matrix)
