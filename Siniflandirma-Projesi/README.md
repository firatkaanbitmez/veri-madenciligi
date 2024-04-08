# Köşe Yazarı Sınıflandırma Projesi

Bu proje, köşe yazarlarının yazılarını sınıflandırmak için bir veri madenciliği yaklaşımını kullanır. Temel amacı, bir yazarın yazı tarzına dayanarak yeni bir köşe yazısının hangi yazar tarafından yazıldığını tahmin etmektir. Bu, metin sınıflandırma problemlerine tipik bir örnektir ve Zemberek kütüphanesi kullanılarak Türkçe metinler üzerinde gerçekleştirilmiştir.

## Proje Açıklaması

Proje adımları şu şekildedir:

1. **Veri Toplama**: Beş farklı köşe yazarına ait en az 20 köşe yazısını içeren bir eğitim kümesi oluşturulmuştur. Her bir yazar için ayrı bir klasörde yazılar bulunmaktadır.

2. **Önişleme Adımları**: Metin temizleme ve kelime köklerine çevirme gibi önişleme adımları gerçekleştirilmiştir. Bu adımlar, metin verilerini işlemeye hazır hale getirmek için yapılmıştır.

3. **Model Oluşturma**: Veri kümesindeki her bir yazının içeriği vektörlere dönüştürülmüş ve bir sınıflandırma modeli oluşturulmuştur. Bu model, yeni bir yazı verildiğinde hangi yazar tarafından yazıldığını tahmin etmek için kullanılacaktır.

4. **Test ve Sonuçlar**: Test verisi kullanılarak modelin başarımı ölçülmüş ve tahmin edilen yazarlar gerçek yazarlarla karşılaştırılmıştır. Bu adımda confusion matrisi gibi istatistiksel analizler yapılmıştır.

## Proje Dosyaları

- **Data Klasörü**: Eğitim verisi için kullanılan köşe yazıları burada bulunmaktadır.
- **Test Klasörü**: Modelin performansını ölçmek için kullanılan test verisi burada bulunmaktadır.
- **Kod Dosyası**: Projenin Python kodları bu dosyada bulunmaktadır.
- **Rapor**: Projenin detaylı raporu ve sonuçlarına bu dosyadan ulaşabilirsiniz.

## Nasıl Çalıştırılır

1. Depoyu klonlayın: `https://github.com/firatkaanbitmez/veri-madenciligi.git`
2. Gerekli kütüphaneleri yükleyin: `pip install Kütüphane_Adi`
3. Kodu çalıştırın: `python3 siniflandirma-projesi.py`

## Gerekli Kütüphaneler

- regex
- numpy
- snowballstemmer
- scikit-learn
- seaborn
- matplotlib

