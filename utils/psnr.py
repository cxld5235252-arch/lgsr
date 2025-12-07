import cv2
import os
import numpy as np

# PSNR计算函数
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

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
    
    # 确保图像数量相同
    assert len(images1) == len(images2), "两个文件夹中的图像数量不相同！"
    print(len(images1))
    psnr_values = []
    for img1, img2 in zip(images1, images2):
        psnr = calculate_psnr(img1, img2)
        psnr_values.append(psnr)
    
    # 计算平均PSNR值
    average_psnr = np.mean(psnr_values)
    print(f"平均PSNR值: {average_psnr}")

if __name__ == "__main__":
    folder1 = "/home/cxl/mycode/diffusion_oppo/ResShift_test/data/test_4k/hr"  # 第一个文件夹路径
    folder2 = "/home/cxl/mycode/diffusion_oppo/ResShift_test/data/test_4k/lr" # 第二个文件夹路径
    main(folder1, folder2)