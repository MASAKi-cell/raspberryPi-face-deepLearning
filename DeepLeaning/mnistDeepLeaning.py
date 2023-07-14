import tensorflow as tf

"""MNISTデータの読み込み"""
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化

"""モデルの定義"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

"""訓練プロセスの定義"""
model.compile(optimizer='adam',  # どうやって学習を最適化するかを決定
              loss='sparse_categorical_crossentropy',  # どうやって損失を定義するかを決定
              metrics=['accuracy'])  # メトリック

"""学習・評価する"""
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
