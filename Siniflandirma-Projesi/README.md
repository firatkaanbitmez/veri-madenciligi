# veri-madenciligi Sınıflandırma Projesi

Proje Detayları
Öncelikle, seçeceğiniz beş farklı köşe yazarına ait  ve her bir yazarın minimum 20 köşe yazısını 
içerecek şekilde bir eğitim kümesi oluşturmanız beklenmektedir. Bu eğitim kümesini kullanarak bir model geliştiriniz. 
Geliştirdiğiniz modele yeni bir köşe yazısı verildiğinde bu yazının hangi yazara ait olduğunu tespit eden veri madenciliği 
projesini gerçekleyiniz. Modeli geliştirirken Zemberek kütüphanesinden yararlanabilirsiniz.
Ödevler teslim edilirken sisteme yüklenecek dosyalar: Rapor ve Kod dosyası
Raporu hazırlarken, model için kullandığınız/oluşturduğunuz veri kümesini, burada ne tür önişleme adımlarını 
gerçekleştirdiğinizi, modeli nasıl oluşturduğunuzu, elde ettiğiniz sonuçların ne olduğunu detaylandırmanız ve yorumlamanız gerekmektedir.


1. Veri Kümesi
İlk olarak, beş farklı köşe yazarının en az 20 köşe yazısını içeren bir eğitim kümesi oluşturuldu. Bu eğitim kümesi, aşağıdaki dizinde bulunmaktadır:

Her yazar için ayrı bir klasör oluşturulmuştur ve her klasörde 20 adet köşe yazısı bulunmaktadır. Yazar klasörleri ve içerdikleri dosyalar aşağıdaki gibidir:
yazar1
doc1.txt
doc2.txt +...
doc20.txt
yazar2
doc1.txt
doc2.txt +...
doc20.txt
yazar3
doc1.txt
doc2.txt +...
doc20.txt
yazar4
doc1.txt
doc2.txt +...
doc20.txt
yazar5
doc1.txt
doc2.txt +...
doc20.txt

2. Önişleme Adımları
Önişleme adımları, her bir köşe yazısının temizlenmesi ve önişleme işlemlerinin uygulanması için gerçekleştirilmiştir. Önişleme adımları aşağıdaki gibidir:

Her bir köşe yazısının içerdiği metinler okunmuş ve temizlenmiştir. Temizleme işlemi sırasında, noktalama işaretleri, sayılar ve özel karakterler kaldırılmıştır.
Temizlenen metinler, her bir kelimenin kök formuna çevrilmiştir. Bu işlem için Zemberek kütüphanesi kullanılmıştır.
Her bir köşe yazısının içerdiği kelimeler, bir kelime haznesine eklenmiştir.

Stopword.txt dosyası veritemizleme kullanmak için C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Siniflandirma-Projesi\stopword.txt


3. Model Oluşturma
Model oluşturma aşamasında, eğitim kümesindeki her bir köşe yazısının içerdiği kelimeler, bir vektör haline getirilmiştir. Vektörler, her bir kelimenin frekansının hesaplanmasıyla oluşturulmuştur. Vektörler, bir matris halinde saklanmıştır.

Model oluşturma aşama sonrasında, her bir yazar için bir vektör elde edilmiştir. Bu vektörler, her bir yazarın yazı stilini temsil etmektedir.

4. Test Verisi
Test verisi, aşağıdaki dizinde bulunmaktadır:


C:\Users\FIRAT\Desktop\myProject\veri-madenciligi\Siniflandirma-Projesi\testyazar

Test verisi, 20 adet köşe yazısından oluşmaktadır. Her bir test dosyası bir yazara aittir(yazar1,yazar2,yazar3,yazar4,yazar5)

5. Sonuçlar
Test verisi, modelin tahmin etmesi için kullanıldıktan sonra, her bir test yazısının tahmin edilen yazarı elde edilmiştir. Tahmin edilen yazarlar, aşağıdaki gibidir:
   test1.txt gerçek yazarı yazar1
    test2.txt gerçek yazarı yazar1
    test3.txt gerçek yazarı yazar1
    test4.txt gerçek yazarı yazar1
    test5.txt gerçek yazarı yazar2
    test6.txt gerçek yazarı yazar2
    test7.txt gerçek yazarı yazar2
    test8.txt gerçek yazarı yazar2
    test9.txt gerçek yazarı yazar3
    test10.txt gerçek yazarı yazar3
    test11.txt gerçek yazarı yazar3
    test12.txt gerçek yazarı yazar3
    test13.txt gerçek yazarı yazar4
    test14.txt gerçek yazarı yazar4
    test15.txt gerçek yazarı yazar4
    test16.txt gerçek yazarı yazar4
    test17.txt gerçek yazarı yazar5
    test18.txt gerçek yazarı yazar5
    test19.txt gerçek yazarı yazar5
    test20.txt gerçek yazarı yazar5
Tahmin edilen yazarlar, istatiksel araştırmalar için kullanılabilir. Örneğin, confusion matrisi oluşturulabilir ve bütün 120 tane köşe yazısının gerçek ve tahmini yazarlarını gösteren bir tablo elde edilebilir.
MÜKKEMEL SONUÇ matrisi
[[4 0 0 0 0]
 [0 4 0 0 0]
 [0 0 4 0 0]
 [0 0 0 4 0]
 [0 0 0 0 4]]    
6. Kod
Projenin kodu, aşağıdaki GitHub deposunda bulunmaktadır:

7. Sonuç
Bu projede, beş farklı köşe yazarının en az 20 köşe yazısını içeren bir eğitim kümesi oluşturulmuştur. Bu eğitim kümesi, önişleme adımlarının uygulanması ve modelin oluşturulması sonrasında, test verisi için kullanılmıştır. Test verisi, modelin tahmin etmesi için kullanılmıştır ve her bir test yazısının tahmin edilen yazarı elde edilmiştir. Tahmin edilen yazarlar, istatiksel araştırmalar için kullanılabilir.

*****************

Bir sorunla karşılaşıyorsanız Muhtemelen Kütüphane eksiktir yada Dosya yolu yanlış olarak algılıyordur. 
Programı masaüstünde Açın yada Dosya yollarını kendinize göre seçin

Varsayılan olarak dosya yolu böyledir. 
data_folder = "./data"
test_data_folder = "./test"
stopwords_path = "./stopword.txt"


Gerekli Kütüphaneler

pip install regex
pip install numpy
pip install snowballstemmer
pip install scikit-learn
pip install seaborn
pip install matplotlib