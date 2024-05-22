from collections import defaultdict
import os
import re
from snowballstemmer import TurkishStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

# Veri klasörü, test verisi klasörü ve stop word dosyası yollarını belirtin
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/data"
test_data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/test"
stopwords_path = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Siniflandirma-Projesi/stopword.txt"

# Stop word dosyasını okuyarak, metinlerdeki gereksiz kelimeleri filtrelemek için bir stop word listesi
with open(stopwords_path, "r", encoding="utf-8") as stopwords_file:
    stop_words = stopwords_file.read().splitlines()

# Eğitim verisi için bir defaultdict oluşturuyoruz. Bu, her yazarın makalelerinin bir listesini saklamak için kullanılacaktır.
#temizlenmiş metin ve her yazar içim dosya adlarını depolamak için bir defaultdict olan trainin_data oluşturuyoruz.
#klasörünlerdeki isimler üzerinde döner klasörün adından yazar adını çıkarır
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
                cleaned_text = re.sub(r'\W', ' ', text)   # Noktalama işaretlerinin kaldırılması (alfasayısal olmayan karakterleri kaldırarak temizler)
                cleaned_text = re.sub(r'\d+', ' ', cleaned_text)  # Sayıların(rakam) kaldırılması 
                cleaned_text = cleaned_text.lower()   # Metnin küçük harflere dönüştürülmesi
                cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])  # Stop words'lerin kaldırılması
                # Türkçe kök bulma işlemi
                stemmer = TurkishStemmer()
                cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()]) #türkçe kelimelerideki kökleri belirlemeke için türkishstmmer kullandık
                # Eğitim verisine temizlenmiş metni ekleyin
                training_data[author_name].append((cleaned_text, file_name))  # Dosya ismiyle birlikte metni ekliyoruz (temizlenmiş metni ve dosya adını ilgili yazar için trainin_data sözlüğüne toplar)

# Eğitim verilerini dengeli hale getirme
# Her yazardan eşit sayıda örnek alın   (DENGELİ EĞİTİM)
min_samples = min(len(texts) for texts in training_data.values())
for author, texts in training_data.items():
    training_data[author] = texts[:min_samples]

# Eğitim ve test verilerini hazırlanması
X = []   #EĞİTİM ÖRNEĞİ
y = []   #ETİKETİ
for author, texts in training_data.items():
    X.extend(text[0] for text in texts)  # Her metinden temizlenmiş metni x listesine ekller
    y.extend([author] * len(texts))      # her metin içinde y listesine Yazar isimi ekler

# Veriyi karıştırma
X, y = shuffle(X, y, random_state=42)   #eğitim verisini rastgeleleştirmek için x y listeleri karıştırılır

# Model oluşturma
model = Pipeline([  # boru hattı oluşturulur.
    ('tfidf', TfidfVectorizer()), #metin verilerini sayısala dönüştürmek için tfidf vektörleyici kullanılır
    ('classifier', SVC())  #metni yazarına göre sınıflandırmak için svc classifier kullanılır
])

# GridSearchCV ile en iyi parametreleri bulma (hiperparametre arama)
#param_grid farklı değişkenleri keşfetmek için kullanıyoruz (parametre_ızgarası)
param_grid = {
    'classifier__C': [0.1, 1, 10, 100], #düzenleme parametresi
    'classifier__gamma': [1, 0.1, 0.01, 0.001], #çekirdek katsayısı
    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'] 
}
# gridSearchCV pipeline tarafından oluşturulur 5kat çapraz doğrulama kullanılarak arama yapmak için
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
#en iyi  gird aramayı yapmak için x y yi rastgele verir

#en iyi modelin eğitilmesi
best_model = grid_search.best_estimator_ #grid_search.best_estimator_ kullanarak grid aramadan en iyi modeli seçer

# Eğitim ve doğrulama verisini ayırmak için train_test_split ayırır kullanarak modeli böler 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train) #en iyi modeli x_Train y_trina eğitim setine göre uydurur

# Model performansını değerlendirme
y_pred = best_model.predict(X_val) # en iyi modeli kullanarak modeli yazarı tahmin eder
print("Sınıflandırma Raporu:")
print(classification_report(y_val, y_pred)) # classification_report kullanarak her yazar için hassasiyer recall f1score ve destek score gösterilir ve yazdırılır



#yeni veri üzerinden modelin Testiii
# Test verisi üzerinde tahminler ve gerçek değerlerin toplanması
true_values = [] #gerçek değerler
predicted_values = [] #tamin

# Test verisindeki her bir yazar için
for author_folder in os.listdir(test_data_folder):
    if os.path.isdir(os.path.join(test_data_folder, author_folder)):
        author_name = author_folder
        # Yazar klasöründeki her bir dosya için
        for file_name in os.listdir(os.path.join(test_data_folder, author_folder)):
            file_path = os.path.join(test_data_folder, author_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                # Metni temizleme ve kök bulma işlemleri
                cleaned_text = re.sub(r'\W', ' ', text)  
                cleaned_text = re.sub(r'\d+', ' ', cleaned_text)  
                cleaned_text = cleaned_text.lower()  
                cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])
                stemmer = TurkishStemmer()
                cleaned_text = ' '.join([stemmer.stemWord(word) for word in cleaned_text.split()])
                # Gerçek yazarı kaydet
                true_values.append(author_name)
                # Model ile tahmin yapma
                prediction = best_model.predict([cleaned_text])[0]
                predicted_values.append(prediction)
                # Dosya ismi ve tahmin edilen yazarı terminalde gösterme
                print("Dosya:", file_name, "- Gerçek Yazar:", author_name, "- Tahmin Edilen Yazar:", prediction)

# Confusion matrix hesaplama
conf_matrix = confusion_matrix(true_values, predicted_values)

# Confusion matrixi görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('5x5 Confusion Matrix')
plt.show()

# Confusion matrixin terminalden çıktı olarak verilmesi
print("5x5 Confusion Matrix:")
print(conf_matrix)
