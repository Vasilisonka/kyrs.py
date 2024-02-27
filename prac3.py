import cv2 as cv
import sys
from matplotlib import pyplot as plt

#1 2
img = cv.imread(cv.samples.findFile("virus_1.jpg"), cv.IMREAD_GRAYSCALE)

laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

#3
img = cv.imread(cv.samples.findFile("virus_1.jpg"))
blur = cv.blur(img,(5,5))
GaussianBlur = cv.GaussianBlur(img,(5,5),0)
# box = cv.boxFilter(img, 0,(7,7), img,(-1,-1), False, cv.BORDER_DEFAULT)

plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(GaussianBlur),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(box),plt.title('Gaussian Blurred')
# plt.xticks([]), plt.yticks([])
plt.show()
