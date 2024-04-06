import os
from zemberek.tokenization import TurkishTokenizer
from zemberek.normalization import TurkishSpellChecker
from zemberek.morphology import TurkishMorphology
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import re

class ColumnBasedPreprocessor:
    def __init__(self, stop_words_file):
        # TurkishTokenizer sınıfını oluştururken bir accepted_type_bits argümanı geçirin
        self.tokenizer = TurkishTokenizer(1)
        
        # TurkishMorphology sınıfını başlatırken kullanarak TurkishSpellChecker sınıfını oluşturun
        morphology = TurkishMorphology.create_with_defaults()
        self.spell_checker = TurkishSpellChecker(morphology)
        
        self.vectorizer = TfidfVectorizer(min_df=1)        
        # Stop word dosyasını oku
        with open(stop_words_file, "r", encoding="utf-8") as f:
            self.stop_words = set(f.read().splitlines())

    def clean_text(self, text):
        return re.sub(r'\W+', ' ', text)

    def lemmatize_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        lemmas = []
        for token in tokens:
            suggestions = self.spell_checker.suggest_for_word(token.content)
            if isinstance(suggestions, list) and suggestions:
                lemmas.append(suggestions[0].word)
            else:
                lemmas.append(token.content)
        return ' '.join(lemmas)

    def preprocess_text(self, text):
        # ... your existing preprocessing steps ...
        text = self.clean_text(text)
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def vectorize_text(self, corpus):
        return self.vectorizer.fit_transform(corpus)

class AuthorClassifier:
    def __init__(self):
        self.model = MultinomialNB()

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def evaluate_model(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict_author(self, X_test):
        return self.model.predict(X_test)

def main():
    # Veri hazırlığı
    corpus = []  # Köşe yazıları burada toplanacak
    labels = []  # Köşe yazarlarının isimleri burada toplanacak

    stop_words_file = "C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Sınıflandırma-Projesi\\stopword.txt"
    preprocessor = ColumnBasedPreprocessor(stop_words_file)
    
    # Veri yollarını belirt
    data_folder = "veri-madenciligi/Sınıflandırma-Projesi/data"
    
    # Tüm yazar klasörlerini dolaş
    for yazar_klasoru in os.listdir(data_folder):
        yazar_klasor_yolu = os.path.join(data_folder, yazar_klasoru)
        # Eğer yazar klasörü bir dizinse
        if os.path.isdir(yazar_klasor_yolu):
            # Yazar klasöründeki tüm belgeleri dolaş
            for dosya in os.listdir(yazar_klasor_yolu):
                dosya_yolu = os.path.join(yazar_klasor_yolu, dosya)
                # Dosyayı oku ve işlemleri yap
                with open(dosya_yolu, "r", encoding="utf-8") as file:
                    for line in file:
                        corpus.append(preprocessor.preprocess_text(line.strip()))
                        labels.append(yazar_klasoru)

    # Veri vektörleştirme
    vectorizer = preprocessor.vectorizer
    X = vectorizer.fit_transform(corpus)

    # Model eğitimi
    classifier = AuthorClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    classifier.model.fit(X_train.toarray(), y_train) # Fit the model with the array representation of X_train

    # Model değerlendirmesi
    accuracy = classifier.evaluate_model(X_test.toarray(), y_test)
    print("Model doğruluğu:", accuracy)

    # Tahminler
    test_corpus = []
    test_labels = []

    for i in range(5):  # Use the actual labels for the test dataset
        test_corpus.append(preprocessor.preprocess_text("test yazısı " + str(i+1)))
        test_labels.append("yazar" + str(i+1))

    X_test = vectorizer.transform(test_corpus)
    predictions = classifier.predict_author(X_test)

    # İstatistiksel analiz
    cm = confusion_matrix(test_labels, predictions, labels=["yazar1", "yazar2", "yazar3", "yazar4", "yazar5"])
    cm_df = pd.DataFrame(cm, index=["yazar1", "yazar2", "yazar3", "yazar4", "yazar5"], columns=["yazar1", "yazar2", "yazar3", "yazar4", "yazar5"])
    print("Confusion Matrix:")
    print(cm_df)

if __name__ == "__main__":
    main()