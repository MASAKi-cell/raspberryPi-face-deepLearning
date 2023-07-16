# from sklearn import model_selection
from PIL import Image
import os
import glob
import numpy as np

classes = ["taimei", "other"]
num_classes = len(classes)
image_size = 50
test_data = 45  # テストデータとトレーニング用のデータを分割する

x_array_train = []  # Numpy配列（train）
x_array_test = []  # Numpy配列（test）
y_label_train = []  # labelを格納（train）
y_label_test = []  # labelを格納（test）


for index, c in enumerate(classes):
    photo_dir = "../test_data/" + c
    files = glob.glob(photo_dir + "/*.jpg")  # pathの生成
    for i, file in enumerate(files):
        if i >= 150:
            break
        image = Image.open(file)
        image = image.convert("RGB")  # RGBに変換
        image = image.resize((image_size, image_size))  # リサイズ
        numpyData = np.asarray(image)  # numpy配列に変換

        if i >= test_data:
            x_array_test.append(numpyData)
            y_label_test.append(index)
        else:  # 訓練用のデータを水増しする
            x_array_train.append(numpyData)
            y_label_train.append(index)

            for angle in range(-20, 20, 5):
                # 画像を回転する
                img_rotate = image.rotate(angle)
                rotate_data = np.asarray(img_rotate)
                x_array_train.append(rotate_data)
                y_label_train.append(index)

                # 画像を反転する
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                transpose_data = np.asarray(img_trans)
                x_array_train.append(transpose_data)
                y_label_train.append(index)

x_array_train = np.array(x_array_train)
x_array_test = np.array(x_array_test)
y_label_train = np.array(y_label_train)
y_label_test = np.array(y_label_test)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(
# x_array_train, y_label_train)  # テストデータと訓練用データを分割

np.save("./x_array_train.npy", x_array_train)  # numpy配列を保存
np.save("./x_array_test.npy", x_array_test)  # numpy配列を保存
np.save("./y_label_train.npy", y_label_train)  # numpy配列を保存
np.save("./y_label_test.npy", y_label_test)  # numpy配列を保存
