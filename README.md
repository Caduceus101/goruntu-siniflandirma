# Görüntü Sınıflandırma  (AlexNet)

Bu proje; **AlexNet** mimarisini kullanan bir derin öğrenme altyapısıdır. 

## Projenin Amacı
Projenin temel amacı, görüntü sınıflandırma iş akışını standardize ederek **ölçeklenebilir (scalable)** bir yapı kurmaktır. Merkezi hiperparametre yönetimi sayesinde, kodun içine müdahale etmeden farklı veri setlerine ve donanım kapasitelerine hızla adapte edilebilir.

##  Teknik Mimari ve Modüller
Proje, temiz kod (clean code) prensiplerine uygun olarak şu bileşenlerden oluşur:

* **`src/config.py`**: Modelin tüm hiperparametrelerini (227x227 giriş boyutu, öğrenme oranı, batch size) tek merkezden yönetir.
* **`src/model.py`**: 5 Konvolüsyonel ve 3 Tam Bağlantılı (Dense) katmandan oluşan, `BatchNormalization` ve `Dropout` ile modernize edilmiş AlexNet tabanlı mimaridir.
* **`src/data_loader.py`**: Görüntülerin klasör yapısından otomatik etiketlenmesini ve `ImageDataGenerator` ile eğitim sürecine hazır hale getirilmesini sağlar.
* **`src/train.py`**: Modelin `Adam` optimizer kullanılarak eğitildiği ve en iyi ağırlıkların (`.h5`) otomatik olarak kaydedildiği ana döngüdür.
* **`src/evaluate.py`**: Eğitilen modelin test seti üzerindeki performansını (Accuracy/Loss) analiz eder ve metrikleri raporlar.
* **`src/utils.py`**: Eğitim sürecindeki başarı ve kayıp grafiklerini görselleştirerek model gelişimini takip etmeyi sağlar.

