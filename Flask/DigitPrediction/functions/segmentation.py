import os
import numpy as np
import cv2
import math


def segmentation(raw_image):
    output_folder = os.path.dirname(os.path.abspath(__file__)) + "/output_images/"
    # Busco los contornos
    mo_image = raw_image.copy()
    _, contours, _ = cv2.findContours(mo_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    '''
    # Muestro los contornos encontrados
    asd = raw_image.copy()
    cv2.drawContours(asd, contours, -1, (0, 255, 0))
    cv2.imshow('draw contours', asd)
    cv2.waitKey(0)
    '''

    # Encuentro los rectangulos que engloban cada contorno encontrado
    maxArea = 0
    rect = []
    for ctr in contours:
        maxArea = max(maxArea, cv2.contourArea(ctr))

    areaRatio = 0.008

    for ctr in contours:
        if cv2.contourArea(ctr) > maxArea * areaRatio:
            rect.append(cv2.boundingRect(cv2.approxPolyDP(ctr, 1, True)))

    # Ordeno todos los rectangulos por su X (Ordeno los contornos encontrados de izquierda a derecha)
    rect.sort(key=lambda b: b[0])


    rect_segmented_image = mo_image.copy()

    all_digits = []
    count = 0
    # Cada uno de los rectangulos contiene un digito.
    # Itero cada rectangulo
    for i in rect:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]

        p1 = (x, y)
        p2 = (x + w, y + h)
        # Segun las coordenadas del rectangulo, extraigo el dígito de la imagen
        digit = raw_image[y:y + h, x:x + w]
        '''
        digit = cv2.GaussianBlur(digit, (3, 3), 0)
        ret, bw_image = cv2.threshold(digit.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Eliminación de residuos - Inicio
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernel)
        digit = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        '''
        # A continuación redimensiono el digito encontrado , teniendo en cuenta la relación de aspecto.
        # Redimensiono la imagen a 28x28 y el digito a 20x20 (Se posiciona el digito en el centro de la imagen)

        o_height = digit.shape[0]
        o_width = digit.shape[1]

        if o_height > o_width:  # Alto > Ancho

            aspectRatio = o_width / (o_height * 1.0)

            height = 20
            width = int(height * aspectRatio)
            digit = cv2.resize(digit, (width, height))


        elif o_width > o_height:  # Ancho > Alto

            aspectRatio = o_height / (o_width * 1.0)

            width = 20
            height = int(width * aspectRatio)

            digit = cv2.resize(digit, (width, height))


        else:  # Alto = Ancho
            digit = cv2.resize(digit, (20, 20))

        cv2.imwrite(output_folder + str(count) + 'beforedeskew.png', digit)
        digit = deskew(digit)
        _, contorno, _ = cv2.findContours(digit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_cnt = contorno[0]
        for digitcontour in contorno:
            if cv2.contourArea(digitcontour) > cv2.contourArea(biggest_cnt):
                biggest_cnt = digitcontour
        rectangulo = cv2.boundingRect(cv2.approxPolyDP(biggest_cnt, 1, True))
        xx = rectangulo[0]
        yy = rectangulo[1]
        ww = rectangulo[2]
        hh = rectangulo[3]
        # Segun las coordenadas del rectangulo, extraigo el dígito de la imagen
        digit = digit[yy:yy + hh, xx:xx + ww]
        rows, cols = digit.shape
        cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        digit = np.lib.pad(digit, (rows_padding, cols_padding), 'constant')

        cv2.imwrite(output_folder + str(count)+'.png', digit)
        count = count + 1
        # Divido los pixeles de la imagen por 255 para normalizarlos en el rango 0-1
        digit = digit / 255
        all_digits.append(digit)

        # Dibujo un rectangulo al rededor del digito.
        cv2.rectangle(rect_segmented_image, p1, p2, 255, 1)

    cv2.imwrite(output_folder + 'segmented.png', rect_segmented_image)

    return all_digits


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
