import os
from zemberek import Zemberek
import re

# Veri kümesinin dizini
data_folder = "C:/Users/FIRAT/Desktop/myProject/veri-madenciligi/Proje-1/zemberek"

# Zemberek nesnesini oluştur
zemberek = Zemberek()

# Tüm yazar klasörlerini al
yazar_klasorleri = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# Her bir yazar için
for yazar_klasoru in yazar_klasorleri:
    # Tüm dosyaları al
    dosya_listesi = os.listdir(yazar_klasoru)
    
    # Her bir dosya için
    for dosya in dosya_listesi:
        dosya_yolu = os.path.join(yazar_klasoru, dosya)
        
        # Dosyayı oku
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            metin = f.read()
        
        # Metni temizle
        metin = re.sub(r'\d+', '', metin)  # Sayıları kaldır
        metin = re.sub(r'[^\w\s]', '', metin)  # Noktalama işaretlerini kaldır
        metin = metin.lower()  # Küçük harfe dönüştür
        
        # Metni köklerine ayır
        kelimeler = zemberek.kelime_cozumle(metin)
        metin = ' '.join([kelime.kok().icerik() for kelime in kelimeler])
        
        # Temizlenen metni dosyaya yaz
        with open(dosya_yolu, 'w', encoding='utf-8') as f:
            f.write(metin)
