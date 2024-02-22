import cv2 as cv
import sys
from matplotlib import pyplot as plt

#1 пункт
img = cv.imread(cv.samples.findFile("bones_1.jpg"))
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
cv.imwrite('D:/2.jpg', img)

#2 пункт
print('Строки, столбцы и каналы:', img.shape) # строки, столбцы и каналы (если изображение цветное)
print('Количество пикселей:',img.size)
print("Тип данных:", img.dtype)
px = img[145, 115]
print(px) #Blue, Green, Red values
px = [155,75,35]
print(px)

#3 пункт
img = cv.imread(cv.samples.findFile("virus_1.jpg"))
height, width = img.shape[:2]
res = cv.resize(img,(3*width, 3*height), interpolation = cv.INTER_CUBIC) # бикубическая интерполяция
cv.imshow("Display window1", res)
k = cv.waitKey(0)
res1 = cv.resize(img,(3*width, 3*height),  interpolation =  cv.INTER_LINEAR )#билинейная интерполяция
cv.imshow("Display window2", res1)
k = cv.waitKey(0)
res2 = cv.resize(img,(3*width, 3*height),  interpolation =  cv.INTER_NEAREST )#билинейная интерполяция
cv.imshow("Display window3", res2)
k = cv.waitKey(0)
w = int(img.shape[1] * 0.5)
h = int(img.shape[0] * 0.5)
res3 = cv.resize(img,(w,h),  interpolation =  cv.INTER_LINEAR )#билинейная интерполяция
cv.imshow("Display window", res3)
k = cv.waitKey(0)

#4 пункт

img1 = cv.imread("virus_1.jpg", cv.IMREAD_GRAYSCALE)
assert img1 is not None, "file could not be read, check with os.path.exists()"
plt.hist(img1.ravel(),256,[0,256]); plt.show()
plt.show()

img = cv.imread(cv.samples.findFile("bones_1.jpg "))
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()