import numpy as np
import cv2
from collections import deque
from keras.models import load_model
model = load_model('emnist_model.h5')

class_mapping = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

def process_image(digit):
    # A continuación redimensiono el digito encontrado , teniendo en cuenta la relación de aspecto.
    # Redimensiono la imagen a 28x28 y el digito a 20x20 (Se posiciona el digito en el centro de la imagen)

    o_height = digit.shape[0]
    o_width = digit.shape[1]

    if o_height == 0:
        digit = np.zeros((30, 30, 1), np.uint8)
        o_height = digit.shape[0]
        o_width = digit.shape[1]

    if o_height > o_width:  # Alto > Ancho

        aspectRatio = o_width / (o_height * 1.0)

        height = 20
        width = int(height * aspectRatio)
        digit = cv2.resize(digit, (width, height))

        # Agrego bordes

        remaining_pixels_w = abs(28 - digit.shape[1])
        add_left = remaining_pixels_w / 2
        add_right = remaining_pixels_w - add_left
        digit = cv2.copyMakeBorder(digit, 0, 0, int(add_left), int(add_right), cv2.BORDER_CONSTANT, value=(0, 0, 0))

        remaining_pixels_h = abs(28 - digit.shape[0])
        add_top = remaining_pixels_h / 2
        add_bottom = remaining_pixels_h - add_top
        digit = cv2.copyMakeBorder(digit, int(add_top), int(add_bottom), 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))



    elif o_width > o_height:  # Ancho > Alto

        aspectRatio = o_height / (o_width * 1.0)

        width = 20
        height = int(width * aspectRatio)

        digit = cv2.resize(digit, (width, height))

        # Agrego bordes para centrar el digito
        remaining_pixels_w = abs(28 - digit.shape[1])
        add_left = round(remaining_pixels_w / 2)
        add_right = round(remaining_pixels_w - add_left)
        digit = cv2.copyMakeBorder(digit, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        remaining_pixels_h = abs(28 - digit.shape[0])
        add_top = int(remaining_pixels_h / 2)
        add_bottom = int(remaining_pixels_h - add_top)
        digit = cv2.copyMakeBorder(digit, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))



    else:  # Alto = Ancho
        digit = cv2.resize(digit, (22, 22))

        remaining_pixels_w = abs(28 - digit.shape[1])
        add_left = int(remaining_pixels_w / 2)
        add_right = int(remaining_pixels_w - add_left)
        digit = cv2.copyMakeBorder(digit, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        remaining_pixels_h = abs(28 - digit.shape[0])
        add_top = int(remaining_pixels_h / 2)
        add_bottom = int(remaining_pixels_h - add_top)
        digit = cv2.copyMakeBorder(digit, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return cv2.resize(digit, (28, 28)).astype('float32') / 255.


def main():
    cap = cv2.VideoCapture(0)

    # define range of blue color in HSV
    lower_blue = np.array([19, 45, 107])
    upper_blue = np.array([51, 134, 236])

    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    results = [0]
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 1000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        image_processed = process_image(digit)
                        image_deep = model.predict(np.reshape(image_processed, (-1, 28, 28, 1)))
                        results = np.argmax(image_deep, axis = 1)
                        print(results)
            cv2.putText(img, "Symbol Identified : " + str(class_mapping[results[0]]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255 ,0, 0), 2)
            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imshow("Main Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break


main()