import tensorflow as tf

import numpy as np

classes = ["taimei", "man"]
num_classes = len(classes)
image_size = 50

# メインの関数を定義する


def main():

    # テストデータANDトレーニング用のデータ取り込み
    X_train = np.load("./x_array_train.npy")
    X_test = np.load("./x_array_test.npy")
    y_train = np.load("./y_label_train.npy")
    y_test = np.load("./y_label_test.npy")

    X_train = X_train.astype("float") / 255  # 正規化
    X_test = X_test.astype("float") / 255  # 正規化
    y_train = tf.keras.utils.to_categorical(
        y_train, num_classes)  # 整数のクラスベクトルから2値クラスの行列への変換
    y_test = tf.keras.utils.to_categorical(
        y_test, num_classes)  # 整数のクラスベクトルから2値クラスの行列への変換

    model = model_train(X_train, y_train)  # モデルのトレーニング
    model_eval(model, X_test, y_test)  # モデルの評価


def model_train(X, y):
    model = tf.keras.models.Sequential()  # モデルの定義
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', input_shape=X.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))  # 活性化関数（relu）の適用
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))  # 過学習の設定

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))  # 全結合層（Dense layer）の追加
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Activation('softmax'))

    opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])  # 損失関数、オプティマイザ、評価指標の指定
    model.fit(X, y, batch_size=32, epochs=100)  # モデルの評価（バッチサイズ32、エポック数100）

    model.save('./taimei_cnn.h5')  # モデルの保存

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])


if __name__ == "__main__":
    main()
