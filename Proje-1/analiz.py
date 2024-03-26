import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
nltk.download('punkt')

def metin_temizle(metin):
    # Noktalama işaretleri ve özel karakterlerin kaldırılması
    temiz_metin = metin.translate(str.maketrans('', '', string.punctuation))
    # Küçük harfe dönüştürme
    temiz_metin = temiz_metin.lower()
    return temiz_metin

def kelime_koklerine_donustur(metin):
    stemmer = PorterStemmer()
    kelimeler = word_tokenize(metin)
    kokler = [stemmer.stem(kelime) for kelime in kelimeler]
    return kokler

def feature_secimi(kokler, en_sik_kelimeler_sayisi):
    frekans_dist = nltk.FreqDist(kokler)
    en_sik_kelimeler = frekans_dist.most_common(en_sik_kelimeler_sayisi)
    return [kelime for kelime, _ in en_sik_kelimeler]

# Dosyadan metni okuyan fonksiyon
def metin_oku(dosya_yolu):
    with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
        return dosya.read()

# Örnek bir metin dosyası yolu
dosya_yolu = 'C:\\Users\\FIRAT\\Desktop\\myProject\\veri-madenciligi\\Proje-1\\doc1.txt'

# Dosyadan metni oku
metin = metin_oku(dosya_yolu)

# Metin temizleme
temiz_metin = metin_temizle(metin)

# Kelime köklerine dönüştürme
kokler = kelime_koklerine_donustur(temiz_metin)

# Feature seçimi
en_sik_kelimeler = feature_secimi(kokler, en_sik_kelimeler_sayisi=1000)

print("En sık kullanılan kelimeler:", en_sik_kelimeler)
