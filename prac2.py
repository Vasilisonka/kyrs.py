import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

# #1 пункт
# img = cv.imread("bones_1.jpg", cv.IMREAD_GRAYSCALE)
# if img is None:
#     sys.exit("Could not read the image.")
# res1, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# res2, thresh2 = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
# res3, thresh3 = cv.threshold(img, 120, 255, cv.THRESH_BINARY)
# res4, thresh4 = cv.threshold(img, 110, 255, cv.THRESH_BINARY)
# res5, thresh5 = cv.threshold(img, 90, 255, cv.THRESH_BINARY)
# res6, thresh6 = cv.threshold(img, 80, 255, cv.THRESH_BINARY)
# titles = ['127','100','120','110','90','80']
# images = [ thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
# plt.show()
#
# #2 пункт
# def connected_component_label(path, con):
#     # Getting the input image
#     img = cv.imread(path, 0)
#     # Converting those pixels with values 1-127 to 0 and others to 1
#     img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
#     # Applying cv2.connectedComponents()
#     num_labels, labels = cv.connectedComponents(img,  connectivity =con)
#
#     # Map component labels to hue val, 0-179 is the hue range in OpenCV
#     label_hue = np.uint8(179 * labels / np.max(labels))
#     blank_ch = 255 * np.ones_like(label_hue)
#     labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
#
#     # Converting cvt to BGR
#     labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
#
#     # set bg label to black
#     labeled_img[label_hue == 0] = 0
#
#     # Showing Original Image
#     plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.title("Orginal Image")
#     plt.show()
#
#     # Showing Image after Component Labeling
#     plt.imshow(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.title("Image after Component Labeling")
#     plt.show()
# connected_component_label("virus_1.jpg", 4 )
# connected_component_label("virus_1.jpg", 8 )

#3 пункт
img = cv.imread("pit.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cv.imshow("Display window", img)
k = cv.waitKey(0)

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img2 = cdf[img]
hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
cv.imshow("Display window", img2)
k = cv.waitKey(0)