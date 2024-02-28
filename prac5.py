import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np

image = cv.imread("bones_4.jpg", 0)
img = cv.imread("bones_4.jpg")
# alpha = 1.0 # Simple contrast control
# beta = 0    # Simple brightness control
# try:
#     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
#     beta = int(input('* Enter the beta value [0-100]: '))
# except ValueError:
#     print('Error, not a number')
# new_image = np.zeros(image.shape, image.dtype)
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for c in range(image.shape[2]):
#             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
# cv.imshow('Original Image', image)
# cv.imshow('New Image', new_image)
# # Wait until user press some key
# cv.waitKey()

def gammaCorrection(img_original, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(img_original, lookUpTable)
    img_gamma_corrected = cv.hconcat([img_original, res])
    cv.imshow("Gamma correction", img_gamma_corrected)

#
# gammaCorrection(img, 0.4)
# cv.waitKey()
# gammaCorrection(img, 0.5)

img = cv.imread("mammo_1.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
template =cv.imread("m1.jpg", cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray,template,cv.TM_CCORR_NORMED)
threshold = 0.999
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv.imwrite('res.png',img)