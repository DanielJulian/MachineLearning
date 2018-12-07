import numpy as np
import math
import cv2
from collections import defaultdict



from keras.models import load_model
model = load_model('emnist_model.h5')

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'


def callback(value):
    pass


def setup_trackbars():
    cv2.namedWindow("Trackbars", 0)
    cv2.createTrackbar("Gauss-Bil", "Trackbars", 0, 1, callback)
    cv2.createTrackbar("Kernel", "Trackbars", 3, 20, callback)
    cv2.createTrackbar("Adapt-Otsu", "Trackbars", 0, 1, callback)
    cv2.createTrackbar("Waste%", "Trackbars", 0, 100, callback)


def get_trackbar_values():
    values = []
    values.append(cv2.getTrackbarPos("Gauss-Bil", "Trackbars"))
    values.append(cv2.getTrackbarPos("Kernel", "Trackbars"))
    values.append(cv2.getTrackbarPos("Adapt-Otsu", "Trackbars"))
    values.append(cv2.getTrackbarPos("Waste%", "Trackbars"))

    return values


def segmentation(raw_image):
    all_digits = []

    try:
        # Busco los contornos
        mo_image = raw_image.copy()
        _, contours, _ = cv2.findContours(mo_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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

            digit = deskew(digit)
            _, contorno, _ = cv2.findContours(digit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contorno) > 0:
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

                count = count + 1
                # Divido los pixeles de la imagen por 255 para normalizarlos en el rango 0-1
                digit = digit / 255
                all_digits.append(digit)

                # Dibujo un rectangulo al rededor del digito.
                cv2.rectangle(rect_segmented_image, p1, p2, 255, 1)
    except:
        pass

    return all_digits


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def filter_binarize_waste_removal(raw_image):
    gauss_bil, kernel, adapt_otsu, waste = get_trackbar_values()
    if kernel % 2 == 0:
        kernel += 1
    try:
        if gauss_bil == 0:
            # Remuevo pequeños ruidos utilizando un filtro gaussiano
            blurred = cv2.GaussianBlur(raw_image, (kernel, kernel), 0)
        else:
            # Remuevo pequeños ruidos utilizando un filtro bilateral
            blurred = cv2.bilateralFilter(raw_image, kernel, kernel*2, kernel/2)

        if adapt_otsu == 0:
            # Binarizo utilizando filtro adaptativo
            bw_image = cv2.adaptiveThreshold(blurred.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                             cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # Binarizo la imagen utilizando Otsu
            ret, bw_image = cv2.threshold(blurred.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


        # Algunas transformaciones morfológicas
        kernel = np.ones((1, 1), np.uint8)
        # bw_image = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, kernel)
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
        mean = mean * waste/100
        for c in contours:
            if mean > cv2.contourArea(c):
                cv2.fillPoly(processed, pts=[c], color=(0, 0, 255))
        processed = cv2.bitwise_and(processed, processed, mask=mask)
        # Eliminación de residuos - Fin
    except:
        processed = raw_image
    return processed


def predict_image(image):
    image_array = np.reshape(image, (1, 28, 28, 1))
    prediction = model.predict(image_array)
    symbol = class_mapping[np.argmax(prediction[0])]
    return str(symbol)


def main():
    setup_trackbars()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    median_prediction = []
    prev = ''
    while cap.isOpened():
        _, img = cap.read()
        img_with_roi = img.copy()
        # Draw lines to indicate region of interest
        # Left Lines
        cv2.line(img_with_roi, (180, 180), (180, 210), (255, 0, 0), 3)
        cv2.line(img_with_roi, (180, 180), (210, 180), (255, 0, 0), 3)

        cv2.line(img_with_roi, (180, 300), (180, 270), (255, 0, 0), 3)
        cv2.line(img_with_roi, (180, 300), (210, 300), (255, 0, 0), 3)

        # Right Lines
        cv2.line(img_with_roi, (440, 180), (470, 180), (255, 0, 0), 3)
        cv2.line(img_with_roi, (470, 180), (470, 210), (255, 0, 0), 3)

        cv2.line(img_with_roi, (470, 300), (440, 300), (255, 0, 0), 3)
        cv2.line(img_with_roi, (470, 300), (470, 270), (255, 0, 0), 3)

        # Extract ROI and show grayscaled
        roi = img[180:300, 180:470]
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grey ROI", gray_image)

        processed = filter_binarize_waste_removal(gray_image)
        cv2.imshow("Processed", processed)

        all_digits = segmentation(processed)
        # Cada 10 frames hago una predicción
        if frame_count == 60:
            prediction = ""
            for digit in all_digits:
                predicted = predict_image(digit)
                prediction += predicted
            median_prediction.append(prediction)
            if len(median_prediction) == 9:
                d = defaultdict(int)
                for i in median_prediction:
                    d[i] += 1
                result = max(d.items(), key=lambda x: x[1])
                median_prediction = []
                prev = result[0]
                cv2.putText(img_with_roi, str(result[0]), (200, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(img_with_roi, str(prev), (200, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            frame_count += 1
        # Show Original with ROI
        cv2.imshow("Primitiva", img_with_roi)
        k = cv2.waitKey(10)
        if k == 27:
            break


main()
