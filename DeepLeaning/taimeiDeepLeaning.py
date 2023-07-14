import tensorflow as tf

# モデルの定義
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
        64, 64, 3)),  # 入力画像のサイズを(64, 64, 3)と仮定
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    # 2クラス分類（特定の人物かどうか）なので、出力は1つ、活性化関数はsigmoid
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # 2クラス分類なので損失関数はbinary_crossentropy
              metrics=['accuracy'])

# モデルの訓練
model.fit(train_images, train_labels, epochs=10)

# モデルの評価
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# モデルの保存
model.save('my_model.h5')
