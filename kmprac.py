import numpy as np
import cv2
import os
from tkinter import Tcl


# import matplotlib.pyplot as plt


def hsv_kmeans(original_image):
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 3
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(img.shape)
    return result_image

    # cv2.imwrite('C:/Users/shubh/Desktop/vision/kmimage.jpg', result_image)


m = 100
n = 10
# myimage = cv2.imread("C:/Users/shubh/Downloads/Shubh_mypy_proj/comp_vision/train/mypic.jpg")
# hsv_kmeans(myimage)

src_dir = "C:/Users/shubh/Desktop/thisone"
# src_dir = "C:/Users/shubh/Downloads/Shubh_mypy_proj/comp_vision/train/traintemp"
dst_dir = "C:/Users/shubh/Desktop/vision"
file_list = os.listdir(src_dir)
file_list = (Tcl().call('lsort', '-dict', file_list))

for i in range(0, m, n):
    for j in range(i, i + 7):
        x = cv2.imread(file_list[j])
        cv2.imwrite('C:/Users/shubh/Desktop/vision/kmimage' + str(i+1) + '.jpg', hsv_kmeans(x))

# def myhistogram(x):
#     x = cv.imread(x)
#     height = x.shape[0]
#     width = x.shape[1]
#     pixels = height * width
#     b, g, r = cv.split(x)
#
#     hist1 = cv.calcHist([b, g, r], [0], None, [pixels], [0, pixels])
#     hist2 = cv.calcHist([b, g, r], [1], None, [pixels], [0, pixels])
#     hist3 = cv.calcHist([b, g, r], [2], None, [pixels], [0, pixels])
#     return hist1, hist2, hist3
