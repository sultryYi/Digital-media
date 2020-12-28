import numpy as np
import cv2

# a = np.linspace(0,9,10)
# a = a.reshape((2,-1))
# print(a)
# print(np.where(a>5)[0][0])

# a = np.random.normal(loc=1, scale=0.5, size=(5, 5))
# print(a)
# b = np.ones((5, 5))
# c = cv2.GaussianBlur(b, (5, 5), 1)
# print(c)

# a = np.array([[1,-1],[3,-4]])
# b = np.array([[1,2],[-3,-4]])
# print(np.sum(a[b==1]))
# print(np.degrees(np.arctan2(a,b)))
# print(np.degrees(np.arctan(b/a)))

# print(np.c_[a,b])
# print(np.r_[a,b])
# print(a)

# a = np.random.normal(loc=0,scale=0.3,size=(7,7))
# a = a-np.min(a)
# print(a)
# print(np.sum(a))

# kernel_size = 3
# sigma = 0
# kx = cv2.getGaussianKernel(kernel_size,sigma)
# ky = cv2.getGaussianKernel(kernel_size,sigma)
# a = np.multiply(kx,np.transpose(ky))
# print(a) 
# print(np.sum(a))
# print(kx)

# a = np.linspace(0,11,12)
# a = a.reshape((4,3))
# b = np.sum(a,axis=1)
# print(b)
# a[0] = [[9,9,9],[9,9,9]]
# print(a)
# print(np.where(b==21))
# a[0][0][:] = np.array([0,0,0])
# b = np.linalg.norm(a)
# print(b)
# a = np.array(a/4,dtype=np.int)
# print(a)
# print(a/b)
# print(np.linalg.norm(a/b))

# img = cv2.imread('../src/1.bmp',cv2.IMREAD_COLOR)
# cv2.line(img,(0,0),(500,500),(133, 255, 116),20)
# cv2.imwrite('../output/test.bmp',img)

def takeThird(elem):
    return elem[2]

a = [([0,1],[0,2],1),([0,2],[9,9],2),([0,9],[3,5],-4)]
a.sort(key=takeThird)
print(a)