import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Retorna la imagen preprocesada (Los digitos en blanco y el fondo negro + corregida la inclinación)
def preprocessing(raw_image):
    output_folder = os.path.dirname(os.path.abspath(__file__)) + "/output_images/"

    # Aplico filtros y binarización
    preprocessed_image = preprocess_image(raw_image)

    # Busco contornos en la imagen para saber cuantos números hay presentes
    _, contours, _ = cv2.findContours(preprocessed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Corrección de inclinación sólo si en la imagen hay mas de 1 número:
    if len(contours) > 1:
        M = get_transformation_matrix(preprocessed_image)  # M es la matriz de transformación
        preprocessed_image = rotate(preprocessed_image, M)

    # Por si se desean observar las imagenes preprocesadas, la guardamos en el disco
    cv2.imwrite(output_folder + 'preprocessed_img.png', preprocessed_image)
    return preprocessed_image


def preprocess_image(raw_image):
    # Remuevo pequeños ruidos utilizando un filtro gaussiano
    # blurred = cv2.GaussianBlur(raw_image, (5, 5), 0)
    # Remuevo pequeños ruidos utilizando un filtro bilateral
    kernel = 5
    blurred = cv2.bilateralFilter(raw_image, kernel, kernel*2, kernel/2)
    # Binarizo la imagen utilizando Otsu
    # ret, bw_image = cv2.threshold(blurred.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Binarizo utilizando filtro adaptativo
    bw_image = cv2.adaptiveThreshold(blurred.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 3)

    # Algunas transformaciones morfológicas
    kernel = np.ones((1, 1), np.uint8)
    bw_image = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, kernel)
    bw_image = cv2.dilate(bw_image, kernel, iterations=1)
    # Eliminación de residuos - Inicio
    # Busco los contornos en la imagen, y calculo el area de cada uno.
    # Calculo una media. Luego remuevo los contornos cuya area este por debajo de la media, ya que es probable que se trate de ruido.
    _, contours, hierarchy = cv2.findContours(bw_image.copy(), 1, 2)
    
    processed = bw_image.copy()
    mask = np.ones(bw_image.shape[:2], dtype="uint8") * 255
    mean = 0
    for c in contours:
        mean += cv2.contourArea(c)
    mean = mean / len(contours)
    mean = mean - mean * 0.60
    for c in contours:
        if mean > cv2.contourArea(c):
            print("Bad contour")
            cv2.fillPoly(processed, pts=[c], color=(0, 0, 255))
    processed = cv2.bitwise_and(processed, processed, mask=mask)

    plt.imshow(processed)

    # Eliminación de residuos - Fin
    return processed


# El input debe ser una imagen binarizada, texto blanco y fondo negro
def get_transformation_matrix(img):

    # Busco todos los pixeles blancos
    pts = cv2.findNonZero(img)

    # Obtengo el rectangulo que engloba los pixeles blancos
    rect = cv2.minAreaRect(pts)

    # rect[0] tiene el centro del rectangulo, rect[1] el ancho y alto, rect[2] el ángulo
    # Dibujo el rectangulo para poder observarlo.
    drawrect = img.copy()
    drawrect = cv2.cvtColor(drawrect, cv2.COLOR_GRAY2BGR)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # box now has four vertices of rotated rectangle
    cv2.drawContours(drawrect, [box], 0, (0, 0, 255), 1)
    cv2.imwrite(os.path.dirname(os.path.abspath(__file__)) + '/output_images/rotated_rect.png', drawrect)

    # Cambio el angulo de rotacion si el ancho es mayor al alto.
    # Esto se hace por la forma en que trabaja cv2.minAreaRect.
    rect = list(rect)
    if rect[1][0] < rect[1][1]:  # Ancho > Alto
        temp = list(rect[1])
        temp[0], temp[1] = temp[1], temp[0]
        rect[1] = tuple(temp)
        rect[2] = rect[2] + 90.0

    # Convierto el rectangulo a un numpy array
    rect = np.asarray(rect)

    # Obtengo la matriz de rotación, que utilizaré posteriormente para rotar la imagen.
    M = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
    return M


def rotate(image, M):
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
