import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np
img = cv.imread("bones_4.jpg", 0)
imp_noise=np.zeros(img.shape,dtype=np.uint8)
cv.randu(imp_noise,0,255)
imp_noise=cv.threshold(imp_noise,245,255,cv.THRESH_BINARY)[1]
in_img=cv.add(img,imp_noise)
median = cv.medianBlur(in_img,5)
plt.subplot(131),plt.imshow(in_img,cmap = 'gray'),plt.title('Шум+картынка')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(imp_noise,cmap = 'gray'),plt.title('Шум')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(median,cmap = 'gray'),plt.title('BOX')
plt.xticks([]), plt.yticks([])
plt.show()

# 2
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

#3