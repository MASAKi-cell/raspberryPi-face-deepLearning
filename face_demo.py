import cv2
import os
import buzzer

import tflite_runtime.interpreter as tflite
import numpy as np

# カスケードファイルパス
cascade_path = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"
)

# CascadeClassifierクラスの生成
cascade = cv2.CascadeClassifier(cascade_path)

# モデルのロード
interpreter = tflite.Interpreter(model_path="./man_cnn.tflite")
interpreter.allocate_tensors()  # テンソルの確保

# インタープリターから入力と出力の詳細を取得
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# カメラモジュール開始
capture = cv2.VideoCapture(0)

desired_width = 1280.0
capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
print(width)

# 検出時のサイズを指定
MIN_SIZE = (300, 300)

while True:
    # 「ESC」を押したら処理を止める、waitKey()はキーボード入力を処理する関数で、引数は入力を待つ時間を指定
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # カメラ画像を読み込む
    _, image = capture.read()

    # 画像反転
    image = cv2.flip(image, -1)

    # OpenCVでグレースケール化、計算処理を高速化するため
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔検出処理
    faces = cascade.detectMultiScale(igray, minSize=MIN_SIZE)

    # 顔検出時以外もフレームを表示させておく
    if len(faces) == 0:
        cv2.imshow('frame', image)
        continue

    for (x, y, w, h) in faces:
        color = (255, 0, 0)
        cv2.rectangle(image, (x, y), (x+w, y+h),
                      color, thickness=8)

        # 顔部分のみを抽出し、モデルの入力サイズにリサイズ
        face_img = image[y:y+h, x:x+w]
        # ここで50x50はモデルが訓練されたときの入力サイズに依存します
        face_img = cv2.resize(face_img, (50, 50))
        face_img = face_img[np.newaxis, ...]  # バッチ次元を追加

        # モデルに入力し、推論を実行
        face_img = face_img.astype("float32") / 255  # 正規化
        interpreter.set_tensor(input_details[0]['index'], face_img)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]['index'])

        # 推論結果が人物に一致する場合、ブザーを鳴らす
        if np.argmax(result) == 1:  # 1は特定の人物に対応するラベル
            print(result, np.argmax(result), "<<< ::: result")
            print("Person A is detected!")
            buzzer.setupBuzzer()
        else:
            print(np.argmax(result), "<<<< ::: 他の人物が検出されました")

    # 顔が検出されたら顔の周りに枠を表示してフレームを表示
    cv2.imshow('frame', image)

capture.release()  # カメラを解放
cv2.destroyAllWindows()  # ウィンドウを破棄
