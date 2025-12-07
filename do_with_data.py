import os
import cv2
import numpy as np

img_list = os.listdir("/home/dongxuan/ctt/datasets_laplacian_tele_warped/train/occ_128/")

for img_name in img_list:
    img = cv2.imread("/home/dongxuan/ctt/datasets_laplacian_tele_warped/train/occ_128/"+img_name)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_image = np.zeros_like(gray_image)
    non_black_mask = gray_image != 0
    img[non_black_mask, :] = [255, 255, 255]
    cv2.imwrite("/home/dongxuan/ctt/datasets_laplacian_tele_warped/train/occ_128/"+img_name, img)  # 保存处理后的图像


   