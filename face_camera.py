import os
import cv2
import buzzer

# カスケードファイルパス
cascade_path = os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"
)

# CascadeClassifierクラスの生成
cascade = cv2.CascadeClassifier(cascade_path)

# カメラモジュールから入力を開始
capture = cv2.VideoCapture(0)

# 検出時のサイズを指定
MIN_SIZE = (150, 150)

# 描画する線の太さ
THICKNESS = 8

while True:
    # 「ESC」を押したら処理を止める、waitKey()はキーボード入力を処理する関数で、引数は入力を待つ時間を指定
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # カメラ画像を読み込む
    _, image = capture.read()

    # raspiのカメラだと反転しているので修正
    # image = cv2.flip(image, -1)

    # OpenCVでグレースケール化して、計算処理を高速化する
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 顔検出を行う部分
    faces = cascade.detectMultiScale(igray, minSize=MIN_SIZE)

    # 顔が検出されたかどうかを判断
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            color = (255, 0, 0)

            # 顔が検出されたら顔の周りに枠を表示してフレームを表示、引数にはウィンドウ名と表示する画像を指定
            cv2.rectangle(image, (x, y), (x+w, y+h),
                          color, thickness=THICKNESS)
            cv2.imshow('frame', image)

            buzzer.setupBuzzer()  # 顔が検出されたらブザーを鳴らす

            print("顔が検出されました")
        else:
            # 顔が検出されなかった場合の処理（顔検出時以外もフレームを表示）
            cv2.imshow('frame', image)
            print("顔が検出されませんでした")


capture.release()  # カメラを解放
cv2.destroyAllWindows()  # ウィンドウを破棄
