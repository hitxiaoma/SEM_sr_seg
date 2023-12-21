# Time: 2023/12/14 17:07
# Author: Yiming Ma
# Place: Shenzhen
from PIL import Image
import os

# 打开原始图像
image = Image.open("4-1-HR-CUT.jpeg")

# 获取裁剪后的图像的尺寸
width, height = image.size

# 设置每个小图像的大小
small_image_size = 256

# 创建rawdata文件夹（如果不存在）
if not os.path.exists("4-1-HR"):
    os.mkdir("4-1-HR")

# 循环裁剪28张小图
k = 0
for i in range(0, width, small_image_size):
    for j in range(0, height, small_image_size):
        # 计算当前小图像的坐标
        left = i
        upper = j
        right = min(i + small_image_size, width)
        lower = min(j + small_image_size, height)

        # 检查小图像是否足够大，如果不够大则跳过
        if right - left < small_image_size or lower - upper < small_image_size:
            continue

        # 裁剪小图像
        small_cropped_image = image.crop((left, upper, right, lower))

        # 保存小图像到rawdata文件夹
        k += 1
        small_cropped_image.save(f"4-1-HR/{k}.jpeg")

# 关闭原始图像
image.close()