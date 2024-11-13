import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_circulos_cuadrante_inferior_derecho(ruta_imagen, max_circulos=None):
    try:
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return

    altura, ancho = imagen.shape[:2]
    mitad_y, mitad_x = altura // 2, ancho // 2

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = gris[mitad_y:altura, mitad_x:ancho]
    gris = cv2.equalizeHist(gris)
    gris = cv2.medianBlur(gris, 5)
    bordes = cv2.Canny(gris, 50, 150)

    circulos = cv2.HoughCircles(
        bordes, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=100, param2=45, minRadio=15, maxRadio=80
    )

    if circulos is not None:
        circulos = np.uint16(np.around(circulos[0, :]))
        if max_circulos is not None:
            circulos = circulos[:max_circulos]

        for x, y, radio in circulos:
            cv2.circle(imagen, (x + mitad_x, y + mitad_y), radio, (0, 255, 0), 2)
            cv2.circle(imagen, (x + mitad_x, y + mitad_y), 2, (0, 0, 255), 3)
    else:
        print("No se detectaron circunferencias en el cuadrante inferior derecho.")

    return imagen

def mostrar_imagen(imagen):
    if imagen is not None:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

ruta_imagen = 'C:\\Users\\acoli\\Pictures\\Captura.jpg'
imagen_circulos = detectar_circulos_cuadrante_inferior_derecho(ruta_imagen, max_circulos=10)
mostrar_imagen(imagen_circulos)
