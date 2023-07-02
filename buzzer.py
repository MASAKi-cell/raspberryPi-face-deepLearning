from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

# BUZZERのピン設定
BUZZER_PIN = 18

# ドレミ - 音名 + オクターブで指定
ONPUS = [
    "C4",   # ド
    "D4",   # レ
    "E4",   # ミ
    "F4",   # ファ
    "G4",   # ソ
    "A4",   # ラ
    "B4",   # シ
    "C5",   # ド
]


def setupBuzzer():

    # 各ピン（GPIO）をbuzzer設定
    factory = PiGPIOFactory()
    buzzer = TonalBuzzer(BUZZER_PIN, pin_factory=factory)

    # 音を鳴らす
    try:
        # 音符指定
        for onpu in ONPUS:
            buzzer.play(Tone(onpu))
            sleep(0.5)
        buzzer.stop()
        sleep(0.5)

        # MIDI note指定
        for note_no in range(60, 80, 5):
            buzzer.play(Tone(midi=note_no))
            sleep(0.5)
        buzzer.stop()
        sleep(0.5)

        # 周波数指定
        for freq in range(300, 400, 100):
            buzzer.play(Tone(frequency=freq))
            sleep(0.5)
        buzzer.stop()
        sleep(0.5)
    except:
        buzzer.stop()
        print("stop")

    return
