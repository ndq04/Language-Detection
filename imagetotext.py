import cv2

img=cv2.imread('data/Anh.png')
print(img.shape)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
