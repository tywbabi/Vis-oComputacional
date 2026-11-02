
import cv2
from ultralytics import YOLO


# 1. Carregar o modelo de inteligencia artificial
modelo = YOLO("yolo11n.pt")

# 2. Inicializar a webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a camera!")
    exit()

print("Sistema de deteccao iniciado!")


# 3. Iniciar a deteccao
while True:

    sucesso, imagem = camera.read()

    if not sucesso:
        print("Erro ao capturar imagem!")
        break

    # 4. Identificar objetos na imagem
    resultados = modelo(imagem, verbose=False)

    # 5. Desenhar os objetos identificados
    imagem_detectada = resultados[0].plot()

    # 6. Mostrar o resultado
    cv2.imshow(
        "Sistema de Visao Computacional",
        imagem_detectada
    )

    # Pressionar ESC para encerrar
    if cv2.waitKey(1) == 27:
        break


# 7. Encerrar o sistema
camera.release()
cv2.destroyAllWindows()

print("Sistema encerrado!")
