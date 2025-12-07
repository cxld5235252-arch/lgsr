import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# SSIM计算函数
def calculate_ssim(img1, img2):
    # 将图片转换为灰度图
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 计算SSIM
    ssim_value = compare_ssim(img1_gray, img2_gray)
    return ssim_value

# 读取图像函数
def read_images(folder):
    images = []
    files = sorted(os.listdir(folder))  # 确保文件是排序的
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        # img = cv2.resize(img, (256, 256))
        images.append(img)
    return images

# 主函数
def main(folder1, folder2):
    images1 = read_images(folder1)
    images2 = read_images(folder2)[:len(images1)]
    print(len(images1))
    # 确保图像数量相同
    assert len(images1) == len(images2), "两个文件夹中的图像数量不相同！"
    
    ssim_values = []
    for img1, img2 in zip(images1, images2):
        ssim = calculate_ssim(img1, img2)
        ssim_values.append(ssim)
    
    # 计算平均SSIM值
    average_ssim = np.mean(ssim_values)
    print(f"平均SSIM值: {average_ssim}")

if __name__ == "__main__":
    folder1 = "/home/cxl/mycode/diffusion_oppo/ResShift_test/data/test_4k/hr"  # 第一个文件夹路径
    folder2 = "/home/cxl/mycode/diffusion_oppo/ResShift_test/data/test_4k/lr" # 第二个文件夹路径
    main(folder1, folder2)