import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussianFilter(m, n, sigma, img):
    gaus_f = np.zeros((m,n))
    
    m = m // 2
    n = n // 2
    x1 = 2*np.pi*(sigma**2)
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaus_f[x+m][y+n] = (1/x1)*x2
            
    #print(gaus_f)
    return convolution(img, gaus_f)
    
def convolution(img, kernel):
    
    h_k = kernel.shape[0]
    w_k = kernel.shape[1]
    
    #padding to retain the original dimensions
    padded_img = np.pad(img, ((h_k // 2, h_k // 2), (w_k // 2, w_k // 2)), 'constant', constant_values = 0)
    
    p_h = padded_img.shape[0]
    p_w = padded_img.shape[1]
    
    h = h_k // 2
    w = w_k // 2
    
    img_conv = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(h, p_h - h):
        for j in range(w, p_w - w):
            x = padded_img[i - h : i - h + h_k, j - w : j-w + w_k]
            x = x.flatten()*kernel.flatten()       
            
            img_conv[i - h][j - w] = x.sum()
    
    img_conv = np.matrix(img_conv)
    plt.imshow(img_conv, cmap = 'gray')
    t = 'gaussian' + str(h_k) + 'x' + str(h_k)
    plt.title(t)
    plt.show()
    unsharp(img, img_conv, h_k)
    high_boost(img, img_conv, h_k)
    

def unsharp(img, blurred, h_k):
    mask = img - blurred
    unsharped = img + mask
    plt.imshow(unsharped, cmap = 'gray')
    t = 'unsharp' + str(h_k) + 'x' + str(h_k)
    plt.title(t)
    plt.show()
    
def high_boost(img, blurred, h_k):
    mask = img - blurred
    boosted = img + 5 * mask
    plt.imshow(boosted, cmap = 'gray')
    t = 'high_boost' + str(h_k) + 'x' + str(h_k)
    plt.title(t)
    plt.show()
    

img = cv2.imread("cc.jpg", 0)
plt.imshow(img, cmap = 'gray')
plt.title('original')
plt.show()

img_1 = gaussianFilter(3, 3, 1, img)
img_2 = gaussianFilter(5, 5, 1, img)
img_2 = gaussianFilter(15, 15, 1, img)
#img_2 = gaussianFilter(30, 30, 1, img)

