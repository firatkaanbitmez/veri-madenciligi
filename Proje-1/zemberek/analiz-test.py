import os
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from zemberek.tokenization import TurkishTokenizer
from zemberek.morphology import TurkishMorphology

# Zemberek kütüphanesi

# Köşe yazarlarının köşe yazılarının bulunduğu dizin
base_dir = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Proje-1\zemberek"

# Her yazar için 20 köşe yazısı alacak şekilde eğitim verisi oluştur
training_data = defaultdict(list)
tokenizer = TurkishTokenizer(accepted_type_bits=1)  # or any other appropriate integer value
morphology = TurkishMorphology.create_with_defaults()

for author_folder in os.listdir(base_dir):
    author_path = os.path.join(base_dir, author_folder)
    if os.path.isdir(author_path):
        author = author_folder
        texts_count = 0
        for file_name in os.listdir(author_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(author_path, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    # Zemberek ile metni işleme
                    tokens = tokenizer.tokenize(text)
                    lemmas = [analysis.dictionary_item.lemma for token in tokens for analysis in morphology.analyze(token.content)]
                    text = " ".join(lemmas)
                    # Check if the processed text is empty or not
                    if text.strip():  # If not empty
                        training_data[author].append(text)
                        texts_count += 1
                        if texts_count >= 20:  # Stop after 20 texts per author
                            break

# Remove authors with less than 20 training documents
authors_to_remove = [author for author, texts in training_data.items() if len(texts) < 20]
for author in authors_to_remove:
    del training_data[author]

# Check if there are documents available for training
if training_data:
    print("Eğitim verisi yüklendi.")
    print("Yazarlar:", list(training_data.keys()))
    print("Toplam köşe yazısı sayısı:", sum(len(texts) for texts in training_data.values()))

    # Veri setini ayırma (Train ve Test)
    X_train, X_test, y_train, y_test = train_test_split(list(training_data.values()), 
                                                        list(training_data.keys()), 
                                                        test_size=0.2, 
                                                        random_state=42)  # Belirli bir rastgele sayı üreteci kullanarak bölme

    # Concatenate the preprocessed text for each author into a single list of strings
    X_train_concatenated = [' '.join(author_texts) for author_texts in X_train]

    # TF-IDF vektörleştirici ve Naive Bayes sınıflandırıcı modeli oluşturma
    stop_words = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
    model = make_pipeline(TfidfVectorizer(stop_words=stop_words), MultinomialNB())

    # Modeli eğitme
    model.fit(X_train_concatenated, y_train)

    # Test verisini kullanarak tahmin yapma
    test_article_processed = preprocess_text(test_article, tokenizer, morphology)
    predicted_author = model.predict([test_article_processed])[0]
    print("Test için Tahmin Edilen Yazar:", predicted_author)

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
else:
    print("Eğitim verisi bulunamadı.")
