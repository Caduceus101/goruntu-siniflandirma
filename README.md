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

src/utils.py: Veri setindeki sınıfların eğitim ve test setleri arasındaki dağılımını görselleştirerek dengeli bir veri yapısı olup olmadığını kontrol 
etmeyi sağlar.


<img width="690" height="625" alt="1000139170" src="https://github.com/user-attachments/assets/2f7fc078-2ddc-4d77-9570-4d1037024449" />
<img width="790" height="812" alt="1000139169" src="https://github.com/user-attachments/assets/3b666a75-27c3-494a-98d0-0bbd5eb5b463" />
<img width="790" height="812" alt="1000139168" src="https://github.com/user-attachments/assets/3b87829d-226a-4e9f-b8a2-63eb28221e70" />
<img width="790" height="812" alt="1000139167" src="https://github.com/user-attachments/assets/84482421-3372-40ff-a663-16e573153946" />

