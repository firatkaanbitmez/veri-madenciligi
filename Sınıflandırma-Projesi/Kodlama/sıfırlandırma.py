# Gerekli kütüphanelerin yüklenmesi
from jpype import startJVM, getDefaultJVMPath, JClass
import os
import string

from jpype import JString

startJVM(getDefaultJVMPath())

# Yolların tanımlanması
ZEMBEREK_PATH = r"C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Sınıflandırma-Projesi\Kodlama\zemberek-full.jar"
DATA_PATH = "data"

# Java sınıflarının tanımlanması
TurkishTokenizer = JClass("zemberek.tokenization.TurkishTokenizer")
TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
TurkishSentenceNormalizer = JClass("zemberek.normalization.TurkishSentenceNormalizer")
Paths = JClass("java.nio.file.Paths")

# Normalizasyon için gerekli objelerin oluşturulması
morphology = TurkishMorphology.createWithDefaults()
tokenizer = TurkishTokenizer.DEFAULT
normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
    Paths.get(str(os.path.join(DATA_PATH, "lm", "lm.2gram.slm"))),
)

# Metnin tanımlanması
text = "'Pek ala, Samara’da 6000 desyatin topragin var ve de 300 atın; e ne olmuş?' Bu soru bni tamamen ele geçirdi ve başka ne düşüneceğimi bilemiyrdum. (Tolstoy)"

# Normalizasyon
normalized_text = str(normalizer.normalize(JString(text)))

# Noktalama işaretlerinin kaldırılması
punctuation_free = "".join([i for i in normalized_text if i not in string.punctuation])

# Sayıların kaldırılması
digit_free = ''.join([i for i in punctuation_free if not i.isdigit()])

# Tokenizasyon
tokens = []
for i, token in enumerate(tokenizer.tokenizeToStrings(JString(digit_free))):
    tokens.append(str(token))

# Stopword eliminasyonu
with open(str(os.path.join(DATA_PATH, "stopwords.txt"))) as file:
    stopwords_zemberek = [line.rstrip() for line in file]

no_stopwords = [i for i in tokens if i not in stopwords_zemberek]

# Lemmatizasyon
analysis_list = [morphology.analyzeAndDisambiguate(JString(word)).bestAnalysis()[0] for word in no_stopwords]
lemm_text = [str(analysis.getDictionaryItem().lemma) for analysis in analysis_list]

# Kök bulma (Stemming)
stem_text = [str(analysis.getStem()) for analysis in analysis_list]

# Sonuçların yazdırılması
print("Normalized Text:", normalized_text)
print("Punctuation Free:", punctuation_free)
print("Digit Free:", digit_free)
print("Tokens:", tokens)
print("No Stopwords:", no_stopwords)
print("Lemmatized Text:", lemm_text)
print("Stemmed Text:", stem_text)
