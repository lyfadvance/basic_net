import numpy as np
import cv2
import matplotlib.pyplot as plt
#140,239,174,232,176,247,142,256
#223,175,244,150,294,183,235,197
#101,193,129,185,132,198,103,204
a = np.array([[[140,239], [174,232], [176,247], [142,256]]], dtype = np.int32)
b = np.array([[[223,175], [244,150], [294,183], [235,197]]], dtype = np.int32)
c=np.array([[[101,193],[129,185],[132,198],[103,204]]],dtype=np.int32)
print(a.shape)
im=cv2.imread('img_3.jpg')
cv2.polylines(im, a, 1, 255)
cv2.polylines(im, b, 1, 255)
cv2.polylines(im, c, 1, 255)
#cv2.fillPoly(im, b, 255)
plt.imshow(im)
plt.show() 
