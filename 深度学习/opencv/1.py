import cv2
import matplotlib
import numpy as np
import  os
img = cv2.imread('rc.jpg')
cv2.imshow("img",img)
cv2.waitKey(0) # 无线时间的显示含糊,1则是显示1ms
cv2.destroyAllWindows() # 按任意键关闭窗口