import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_lineas(ruta_imagen, umbral_borde1=50, umbral_borde2=150, umbral_hough=200, max_lineas=None):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta: {ruta_imagen}")
    
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, umbral_borde1, umbral_borde2, apertureSize=3)
    
    lineas = cv2.HoughLines(bordes, 1, np.pi / 180, umbral_hough)
    if lineas is None:
        print("No se detectaron líneas.")
        return img
    
    contador_lineas = 0
    for linea in lineas:
        if max_lineas is not None and contador_lineas >= max_lineas:
            break
        rho, theta = linea[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        contador_lineas += 1
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return img

ruta_imagen = 'C:\\Users\\acoli\\Pictures\\Captura.jpg'
detectar_lineas(ruta_imagen, max_lineas=8)
