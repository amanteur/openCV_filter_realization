import numpy as np
import cv2
import time

def bf_cpp (img_in, d, sigma_color, sigma_space, borderType):
    h, w, ch = img_in.shape

    if (sigma_color <= 0 ):
        sigma_color = 1
    if (sigma_space <= 0 ):
        sigma_space = 1

    gauss_color_coeff = -0.5/(sigma_color*sigma_space)
    gauss_space_coeff = -0.5/(sigma_space*sigma_color)

    if (d <= 0):
        radius = int(sigma_space * 1.5)
    else:
        radius = d/2
    radius = max(radius,1)
    d = radius * 2 + 1


def filter_bilateral( img_in, win_diam, sigma_s, sigma_v, reg_constant= 1e-8 ):

    # функция гаусса
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3)*1.0/3.0

    # размер окна
    if (win_diam <= 0):
        win_width = int( 1.5 * sigma_s)
    else:
        win_width = win_diam // 2
    win_width = max(win_width, 1)
    win_diam = win_width * 2 + 1

    # весовые коэффициенты
    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    # проход по картинке
    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # подсчитываем пространственный вес
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # подсчитываем вес радиометрический
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # сохраняем резалт
            result += off*tw
            wgt_sum += tw

    # нормализация результата
    return result/wgt_sum


if __name__ == '__main__':

    #читаем файл
    I = cv2.imread('polymem.jpg').astype(np.float32)

    # openCV
    opencv_bf = cv2.bilateralFilter(I, 16, 50, 50)
    cv2.imwrite("i_bil_cv.png", opencv_bf)

    # свой
    tic = time.time()
    own_bf = np.stack([
        filter_bilateral(I[:, :, 0], 16, 5, 25),
        filter_bilateral(I[:, :, 1], 16, 5, 25),
        filter_bilateral(I[:, :, 2], 16, 5, 25)], axis=2)
    toc = time.time()

    # запись в файл
    cv2.imwrite('i_bil_own.png',own_bf)
    cv2.imwrite('diff.png', (own_bf - opencv_bf))
    print('Elapsed time: %f sec.' % (toc - tic))
