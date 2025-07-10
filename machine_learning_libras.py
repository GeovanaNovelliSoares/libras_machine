import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import traceback

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Erro: Não foi possível acessar a câmera.")

    hands = mp.solutions.hands.Hands(max_num_hands=1)

    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    try:
        model = load_model('keras_model.h5')
        print("Modelo carregado")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    word = ''
    last_prediction = ''
    frames_count = 0
    stable_frames_required = 15

    while True:
        success, img = cap.read()
        if not success:
            print("Erro ao capturar imagem")
            break

        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape

        if handsPoints is not None:
            for hand in handsPoints:
                x_max, y_max = 0, 0
                x_min, y_min = w, h

                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    x_min = min(x, x_min)
                    y_max = max(y, y_max)
                    y_min = min(y, y_min)

                padding = 50
                x1 = max(x_min - padding, 0)
                y1 = max(y_min - padding, 0)
                x2 = min(x_max + padding, w)
                y2 = min(y_max + padding, h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                try:
                    imgCrop = img[y1:y2, x1:x2]
                    if imgCrop.size == 0:
                        raise ValueError("Recorte vazio — coordenadas fora do frame.")

                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data[0] = normalized_image_array

                    prediction = model.predict(data, verbose=0)
                    indexVal = np.argmax(prediction)
                    current_letter = classes[indexVal]

                    if current_letter == last_prediction:
                        frames_count += 1
                    else:
                        frames_count = 0
                        last_prediction = current_letter

                    if frames_count == stable_frames_required:
                        if current_letter == 'ACABOU':
                            print(f"Palavra formada: {word}")
                            word = ''
                        else:
                            word += current_letter
                            print(f"Palavra parcial: {word}")
                        frames_count = 0

                    cv2.putText(img, current_letter, (x1, y1 - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)

                except Exception as e:
                    print(f"[Erro] Processamento da imagem: {e}")
                    continue

        cv2.putText(img, f'Palavra: {word}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.imshow('Imagem', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nExceção não tratada capturada:")
        traceback.print_exc()
