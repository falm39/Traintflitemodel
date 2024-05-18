**CNN ve TensorFlow ile Duygu Tanıma**

Bu proje, FER-2013 veri setini kullanarak gri tonlamalı yüz resimlerinden duyguları tanımak için bir Konvolüsyonel Sinir Ağı (CNN) eğitir. Eğitilen model daha sonra uç cihazlarda kullanılmak üzere TensorFlow Lite (TFLite) formatına dönüştürülür.

**Gereksinimler**

Python 3.x
TensorFlow 2.x
Keras
Pillow (resim işleme için)

**Kurulum**

Gerekli paketleri yüklemek için:

```
pip install tensorflow keras pillow
```
**Veri Seti**

FER-2013 veri seti, yüz ifadelerini yedi duyguya göre kategorize eder: öfke, iğrenme, korku, mutluluk, üzüntü, şaşkınlık ve nötr. Veri setinin aşağıdaki gibi düzenlendiğinden emin olun:

```
fer2013/
    train/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
    test/
        angry/
        disgust/
        fear/
        happy/
        neutral/
        sad/
        surprise/
```

Kullanım
Veri Hazırlığı:
Eğitim ve test veri dizinleri betikte ayarlanmalıdır:

```
train_dir = '/path/to/fer2013/train'
test_dir = '/path/to/fer2013/test'
```
Model Eğitimi ve Dönüştürme:
CNN modelini eğitmek ve TFLite formatına dönüştürmek için betiği çalıştırın:

```
python3 train.py
```
Kod Açıklaması
Veri Artırma ve Yükleme:

```
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode='categorical')
```
Model Tanımı:
Conv2D ve MaxPooling2D katmanlarına sahip basit bir CNN modeli tanımlanır:

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
```
Model Eğitimi:
Model derlenir ve eğitilir:

```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=30, validation_data=test_generator)
```
Model Dönüştürme:
Eğitilen model TFLite formatına dönüştürülür:

```
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```
Sonuçlar
Betik çalıştırıldıktan sonra, gerçek zamanlı duygu tanıma için uç cihazlarda kullanılabilecek bir model.tflite dosyanız olacak.

Kaynaklar
TensorFlow belgeleri: https://www.tensorflow.org/
Keras belgeleri: https://keras.io/
