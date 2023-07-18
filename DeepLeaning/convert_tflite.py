import tensorflow as tf
h5_model = tf.keras.models.load_model('taimei_cnn.h5')  # モデルのロード
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)  # tfliteに変換
tflite_model = converter.convert()

with open('taimei_cnn.tflite', 'wb') as f:
    f.write(tflite_model)
