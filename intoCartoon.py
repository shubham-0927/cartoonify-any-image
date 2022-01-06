import cv2
import numpy as np
import matplotlib.image as img
from matplotlib import pyplot as plt
# image is numpy array of many dimensions
img = cv2.imread("RDJ.jpg")
# create Edge mask


def edge_mask(img, line_size,blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


line_size = 7
blur_value = 7
edges = edge_mask(img, line_size, blur_value)
# color quantization
# k value determines the number of colours in the image
total_color = 7
k = total_color
# Transer the image
data = np.float32(img).reshape(-1,3)
# determine criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
# implementing k-Means
ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape(img.shape)
blurred = cv2.bilateralFilter(result, d=10, sigmaColor=100, sigmaSpace=250)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
filename = 'cartoon.jpg'
cv2.imwrite(filename, cartoon)