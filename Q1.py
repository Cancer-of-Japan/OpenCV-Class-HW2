import numpy as np
import cv2
import matplotlib.pyplot as plt

#<<< Load image and preprocess >>>#
img = cv2.imread('Q1_Image/coin01.jpg')
#plt.imshow(img)
plt.show()
#Turn to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(img_gray)
plt.show()
#Apply Gaussian filter
img_gray_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
#plt.imshow(img_gray_blur)
plt.show()

#<<< Drawing part >>>#
ret, thresh = cv2.threshold(img_gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(type(contours))
print(len(contours))
print(type(hierarchy))
print(hierarchy)


im_con = img.copy()
cv2.drawContours(im_con, contours, -1, (0, 255, 0), 2)
cv2.imwrite('coin1_result.png', im_con)
plt.imshow(im_con)
plt.show()
print(len(contours))
