from PIL import Image
from sklearn import cross_validation
import os
import glob
import numpy as np

classes = ["taimei"]
num_classes = len(classes)
image_size = 50

numpy_array = []  # Numpy配列
numpy_label = []  # labelを格納

for index, c in enumerate(classes):
    photo_dir = "./" + c
    files = glob.glob(photo_dir + ".jpg")  # pathの生成
    for i, file in enumerate(files):
        if i >= 100:
            break
        image = Image.open(file)
        image = image.convert("RGB")  # RGBに変換
        image = image.resize((image_size, image_size))  # リサイズ
        numpyData = np.asarray(image)  # numpy配列に変換
        numpy_array.append(numpyData)
        numpy_label.append(index)

numpy_array = np.array(numpy_array)
numpy_label = np.array(numpy_label)
