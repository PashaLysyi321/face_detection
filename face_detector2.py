# Импорт необходимых модулей
from imutils.video import VideoStream, FPS
import cv2
import numpy as np
import dlib

# Запуск видео потока
vs = VideoStream(src=0).start()

# Подключение детектора, настроенного на поиск человеческих лиц
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/lysyi/Desktop/face_detection/shape_pred_68.dat')

while True:
 
    # Получение изображения из видео потока
    frame = vs.read()

    # Конвертирование изображения в черно-белое
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц и построение прямоугольного контура
    faces = detector(grayFrame)

    # Обход списка всех лиц попавших на изображение
    for face in faces:

        # Выводим количество лиц на изображении
        cv2.putText(frame, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Получение координат вершин прямоугольника и его построение на изображении
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
 
        # Получение координат контрольных точек и их построение на изображении
        landmarks = predictor(grayFrame, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    cv2.putText(frame, "Press ESC to close frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Вывод преобразованного изображения
    cv2.imshow("Frame", frame)

    # Для выхода из цикла нажать ESC
    key = cv2.waitKey(1)
    if key == 27:
        break