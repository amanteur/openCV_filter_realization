import cv2
import numpy as np

def conv(src, kernel):
    # читаем файл
    I = cv2.imread(src).astype(np.float32)

    # openCV
    opencv_conv = cv2.filter2D(src=I,ddepth=255,kernel=kernel)

    # запись
    cv2.imwrite("polymem_changed.png", opencv_conv)

source = 'polymem.jpg'
kernel = np.array([[0,0,0],[0,-1,0],[0,0,0]])
conv(src= source, kernel= kernel)