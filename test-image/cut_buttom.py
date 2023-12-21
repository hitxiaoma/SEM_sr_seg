# Time: 2023/12/14 17:01
# Author: Yiming Ma
# Place: Shenzhen
import os
from PIL import Image

# 定义输入和输出文件夹路径
filename = "4-1-HR.jpeg"
output_folder = "4-1-HR-CUT.jpeg"

# 遍历小文件夹中的所有图像文件

base_name = os.path.splitext(filename)[0]

# 打开原始图像
image = Image.open(filename)

# 裁剪底部75行像素
cropped_image = image.crop((0, 0, image.width, image.height - 150))

cropped_image.save(output_folder)

# 关闭原始图像
image.close()