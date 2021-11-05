import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.jpg", 0)
print(img.shape)


f = cv2.getGaussianKernel(15, 1)
kernal = f.dot(np.transpose(f))
print(kernal)
blur = cv2.GaussianBlur(img, (15,15), 1)
plt.imshow(blur, cmap = 'gray')
plt.title("openCV Gaussian 5 x 5")
plt.show()
mask = img - blur
ub = img + mask
hb = img + 2*mask
plt.imshow(ub, cmap = 'gray')
plt.title("ub")
plt.show()
plt.imshow(hb, cmap = 'gray')
plt.title("hb")
plt.show()