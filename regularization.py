import numpy as np
import cv2
from contourprocess import regularization


ori_img1 = cv2.imread('./test.jpg')
# 中值滤波，去噪
ori_img = cv2.medianBlur(ori_img1, 5)
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
ret, ori_img = cv2.threshold(ori_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=8)

regularization_contours = []
# 遍历联通域
for i in range(1, num_labels):
    img = np.zeros_like(labels)
    index = np.where(labels==i)
    img[index] = 255
    img = np.array(img, dtype=np.uint8)

    regularization_contour =regularization.boundary_regularization(img).astype(np.int32)
    regularization_contours.append(regularization_contour)
    
    single_out = np.zeros_like(ori_img1)
    cv2.polylines(img=single_out, pts=[regularization_contour], isClosed=True, color=(255, 0, 0), thickness=3)
    cv2.imwrite('./result/single_out_{}.jpg'.format(i), single_out)



cv2.polylines(img=ori_img1, pts=regularization_contours, isClosed=True, color=(255, 0, 0), thickness=3)
cv2.imwrite('all_out.jpg', ori_img1)