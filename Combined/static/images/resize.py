import cv2
img = cv2.imread('AI_BGcopy2.jpg')
img = cv2.resize(img,(1460,700))
cv2.imwrite('test.png',img)
