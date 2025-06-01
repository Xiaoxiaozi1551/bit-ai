import cv2
import numpy as np


def laplacian_pyramid_blending(img1, img2, levels=6):
    # 生成高斯金字塔
    img1_gauss = [img1]
    img2_guass = [img2]
    img_blend_pyramid = []

    for level in range(levels):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        # cv2.imshow('img' + str(level), img1)
        # cv2.imshow('img' + str(level), img2)
        img1_gauss.append(img1)
        img2_guass.append(img2)
        
    # 生成拉普拉斯金字塔
    img1_pyramid = [img1_gauss[levels-1]]
    img2_pyramid = [img2_guass[levels-1]]
    for i in range(levels-1, 0, -1):
        GE1 = cv2.pyrUp(img1_gauss[i])
        GE2 = cv2.pyrUp(img2_guass[i])
        L1 = cv2.subtract(img1_gauss[i-1], GE1)
        L2 = cv2.subtract(img2_guass[i-1], GE2)
        # cv2.imshow('L1_'+str(i), L1)
        # cv2.imshow('L2_' + str(i), L2)
        img1_pyramid.append(L1)
        img2_pyramid.append(L2)

    # 从底层到顶层进行融合
    LS = []
    for l1, l2 in zip(img1_pyramid, img2_pyramid):
        rows, cols, dpt = l1.shape
        # ls = (l1+l2)//2
        ls = np.hstack((l1[:, 0:cols//2], l2[:, cols//2:]))
        # ls = l1.copy()
        # ls[0:50, 0:50] = l2[0:50, 0:50]
        # ls[0:rows//2, 0:cols//2] = l2[0:rows//2, 0:cols//2]
        LS.append(ls)

    # 重建融合图像
    img_blend = LS[0]
    for i in range(1, levels):
        img_blend = cv2.pyrUp(img_blend)
        img_blend = cv2.add(img_blend, LS[i])
    return img_blend


# 读取两幅图像
img1 = cv2.imread('3.jpeg')
img2 = cv2.imread('1.jpg')

# 调整图像大小，使其具有相同的尺寸
img1 = cv2.resize(img1, (704, 512))
img2 = cv2.resize(img2, (704, 512))

# 融合图像
blended_img = laplacian_pyramid_blending(img1, img2)

# 显示融合结果
cv2.imshow('Blended Image', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()