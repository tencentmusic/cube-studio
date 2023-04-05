import random

import cv2
import numpy as np

# 读取图片
color = (0, 0, 255)
img = cv2.imread('example.jpg')
save_path = 'example1.jpg'
# print(mask)

# print(mask)
for i in range(10):
    color = random.choice([(0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)])
    mask = np.random.randint(0, 2, (img.shape[0], img.shape[1]))
    mask_img = mask*list(color)

    # mask_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #
    # for y,row in enumerate(mask):
    #     for x,item in enumerate(row):
    #         # print(x, item)
    #         # print(mask[y,x])
    #         if item:
    #             # print(mask_img[x,y])
    #             mask_img[y,x] = list(color)  # 设定为红色
    #         else:
    #             mask_img[y, x] = [0, 0, 0]

    print(mask_img)
    print(mask_img.shape)

    mask_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    mask_img[:,:] = [0, 0, 255]  # 设定为红色
    # print(mask_img)
    # print(mask_img.shape)

    alpha = 0.5  # 设定透明度
    mask_img = cv2.addWeighted(img, alpha, mask_img, 1 - alpha, 0)

    # 在图片上绘制 mask
    img = cv2.addWeighted(img, 1, mask_img, 0.5, 0)

cv2.imwrite(save_path, img)


#
# # 读取图片
# # 创建半透明的 mask 矩阵
# mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
# # print(mask)
# mask[:,:] = [0, 0, 255]  # 设定为红色
# print(mask)
# alpha = 0.5  # 设定透明度
# mask = cv2.addWeighted(img, alpha, mask, 1 - alpha, 0)
#
# # 在图片上绘制 mask
# result = cv2.addWeighted(img, 1, mask, 0.5, 0)
# cv2.imwrite(save_path, result)
