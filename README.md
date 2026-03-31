Görüntü Sınıflandırma (AlexNet & PyTorch)

Bu proje; PyTorch kütüphanesi ve AlexNet mimarisini kullanan, yüksek çözünürlüklü görseller üzerinde çalışmak üzere optimize edilmiş bir derin öğrenme altyapısıdır.

Projenin Amacı
Görüntü sınıflandırma iş akışını standardize ederek ölçeklenebilir (scalable) bir yapı kurmaktır. Merkezi hiperparametre yönetimi sayesinde, kodun içine müdahale etmeden farklı veri setlerine ve donanım kapasitelerine hızla adapte edilebilir.

Teknik Mimari ve Modüller
Proje, temiz kod (clean code) prensiplerine uygun olarak şu modüllerden oluşur:

src/config.py: Modelin tüm hiperparametrelerini (Örn: 1024x1024 giriş boyutu, Batch Size) tek merkezden yönetir ve donanım (CPU/GPU) seçimini otomatik yapar.

src/model.py: PyTorch'un önceden eğitilmiş AlexNet modelini temel alır. Transfer Learning yaklaşımıyla, son tam bağlantılı katman (FC) projedeki sınıf sayısına göre dinamik olarak yeniden yapılandırılmıştır.

src/data_loader.py: Verilerin klasör yapısından (ImageFolder) otomatik yüklenmesini, 1024 boyutuna ölçeklenmesini ve normalizasyon işlemlerini gerçekleştirir.

src/train.py: Modelin Adam optimizasyon algoritması ve CrossEntropyLoss kullanılarak eğitildiği ana döngüdür. Eğitim sonunda ağırlıkları .pth formatında kaydeder.

src/evaluate.py: Eğitilen modelin test seti üzerindeki performansını Accuracy ve detaylı Sınıflandırma Raporu (Precision, Recall, F1-Score) ile analiz eder.

src/utils.py: Veri setindeki sınıfların eğitim ve test setleri arasındaki dağılımını görselleştirerek dengeli bir veri yapısı olup olmadığını kontrol etmeyi sağlar.
