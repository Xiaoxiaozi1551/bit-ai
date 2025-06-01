import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def gauss_convolve(filter, img, padding, strides=1):
    print(1)
    height, width = img.shape
    f_height, f_width = filter.shape
    padding_height, padding_width = padding
    stride_height, stride_width = strides

    # 计算卷积的尺寸
    output_height = (height - f_height + 2 * padding_height) // stride_height + 1
    output_width = (width - f_width + 2 * padding_width) // stride_width + 1

    # 对输入图像进行填充
    padded_img = np.pad(img, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    # 初始化卷积结果
    convolved = np.zeros((output_height, output_width))

    # 进行卷积操作
    for h in range(output_height):
        for w in range(output_width):
            window = padded_img[h * stride_height:h * stride_height + f_height,
                     w * stride_width:w * stride_width + f_width]
            convolved[h, w] = np.sum(window * filter)
    print(convolved.shape)

    return convolved

def undersampling(img):
    print(2)
    return img[::2, ::2]

def gaussian_kernel(sigma, dim):
    temp = np.arange(dim) - (dim // 2)
    assistant = np.tile(temp, (dim, 1))
    temp = 2 * sigma * sigma
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + assistant.T ** 2) / temp)
    return result

def build_gaussian_pyramid(image, num_octaves, num_scales, sigma0):
    SIGMA = 1.6
    SIGMA_INIT = 0.5
    k = 2 ** (1/n)
    sigma = []
    sample = []
    for i in range(num_octaves):
        temp = []
        if i == 0:
            sample.append(image)
        else:
            sample.append(undersampling(sample[-1]))
        for j in range(num_scales):
            temp.append((k ** j) * sigma0 * (2 ** i))
        sigma.append(temp)

    pyramid = []
    for i in range(num_octaves):
        octave = []
        for j in range(num_scales):
            dim = int(6 * sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            print('dim = ', dim)
            octave.append(gauss_convolve(gaussian_kernel(sigma[i][j], dim), sample[i], [dim // 2, dim // 2], [1, 1]))
        pyramid.append(octave)

    # 差分金字塔
    DoG_Pyramid = []
    for i in range(num_octaves):
        temp = []
        for j in range(num_scales - 1):
            temp.append(pyramid[i][j + 1] - pyramid[i][j])
        DoG_Pyramid.append(temp)
    return DoG_Pyramid, pyramid


def detect_keypoints(pyramid, threshold=0.5, edge_threshold=10):
    keypoints = []

    num_octaves = len(pyramid)
    num_scales = len(pyramid[0])


    for octave_idx in range(num_octaves):
        octave = pyramid[octave_idx]

        for scale_idx in range(1, num_scales - 1):
            prev_scale = octave[scale_idx - 1]
            curr_scale = octave[scale_idx]
            next_scale = octave[scale_idx + 1]

            dog = curr_scale

            # 对DoG进行非极大值抑制
            mask = np.logical_and(dog > threshold, dog > np.roll(dog, 1, axis=0))
            mask = np.logical_and(mask, dog > np.roll(dog, -1, axis=0))
            mask = np.logical_and(mask, dog > np.roll(dog, 1, axis=1))
            mask = np.logical_and(mask, dog > np.roll(dog, -1, axis=1))
            mask = np.logical_and(mask, np.abs(dog) > edge_threshold)
            keypoints_octave = np.argwhere(mask)

            for keypoint in keypoints_octave:
                i, j = keypoint
                pixel_value = dog[i, j]
                neighbors = dog[i-1:i+2, j-1:j+2]

                if neighbors.size > 0 and (pixel_value == np.max(neighbors) or pixel_value == np.min(neighbors)):
                    dx = 0.5 * (dog[i, j+1] - dog[i, j-1])
                    dy = 0.5 * (dog[i+1, j] - dog[i-1, j])
                    ds = 0.5 * (next_scale[i, j] - prev_scale[i, j])

                    dxx = dog[i, j+1] - 2 * dog[i, j] + dog[i, j-1]
                    dyy = dog[i+1, j] - 2 * dog[i, j] + dog[i-1, j]
                    dss = next_scale[i, j] - 2 * curr_scale[i, j] + prev_scale[i, j]
                    dxy = 0.25 * (dog[i+1, j+1] - dog[i+1, j-1] - dog[i-1, j+1] + dog[i-1, j-1])
                    dxs = 0.25 * (next_scale[i, j+1] - next_scale[i, j-1] - prev_scale[i, j+1] + prev_scale[i, j-1])
                    dys = 0.25 * (next_scale[i+1, j] - next_scale[i-1, j] - prev_scale[i+1, j] + prev_scale[i-1, j])

                    D = np.array([[dxx, dxy, dxs],
                                  [dxy, dyy, dys],
                                  [dxs, dys, dss]])

                    offset = -np.linalg.inv(D) @ np.array([dx, dy, ds])

                    if np.abs(offset[0]) < 0.5 and np.abs(offset[1]) < 0.5 and np.abs(offset[2]) < 0.5:
                        x = (j + offset[0]) * (2 ** octave_idx)
                        y = (i + offset[1]) * (2 ** octave_idx)
                        scale = sigma * (2 ** (octave_idx + scale_idx / num_scales))
                        keypoints.append((x, y, scale))

    # 执行关键点重复数据删除
    keypoints = list(set(keypoints))

    print(f"Number of keypoints: {len(keypoints)}")

    return keypoints


def compute_descriptors(image, keypoints, descriptor_size=16, orientation_bins=8):
    descriptors = []

    for keypoint in keypoints:
        x, y, scale = keypoint

        # 根据默认比例计算描述符窗口的大小
        window_size = descriptor_size

        # 使用默认方向 0
        orientation = 0

        # 计算关键点的方向直方图
        histogram = compute_orientation_histogram(image, x, y, window_size, orientation, orientation_bins)

        # 使用圆形邻域平滑直方图
        histogram = smooth_histogram(histogram)

        # 通过连接子直方图创建描述符
        descriptor = create_descriptor(histogram)

        # 将描述符标准化为单位长度
        descriptor /= np.linalg.norm(descriptor)

        descriptors.append(descriptor)

    return descriptors


def compute_orientation_histogram(image, x, y, window_size, orientation, orientation_bins):
    # 计算图像的梯度大小和方向
    gradients = compute_gradients(image, x, y)
    magnitudes = np.sqrt(gradients[:, :, 0] ** 2 + gradients[:, :, 1] ** 2)
    orientations = np.arctan2(gradients[:, :, 1], gradients[:, :, 0])

    # 计算关键点窗口的方向直方图
    histogram = np.zeros((window_size * 2 + 1, window_size * 2 + 1, orientation_bins))
    bin_width = 2 * np.pi / orientation_bins

    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            # 计算相对于关键点的像素坐标
            px = x + i
            py = y + j

            if px < 0 or px >= orientations.shape[0] or py < 0 or py >= orientations.shape[1]:
                continue

            # 计算像素与关键点之间的方向差
            orientation_diff = orientations[int(px), int(py)] - orientation

            # 将方向差异包装到范围 [0, 2*pi]
            orientation_diff = (orientation_diff + 2 * np.pi) % (2 * np.pi)

            # 计算方向差异的 bin 索引
            bin_index = int(orientation_diff / bin_width)

            # 用梯度幅值更新直方图箱
            histogram[i + window_size, j + window_size, bin_index] += magnitudes[int(px), int(py)]
    # 归一化直方图
    norm = np.linalg.norm(histogram)
    histogram /= max(norm, np.finfo(float).eps)

    return histogram


def compute_gradients(image, x, y):
    # 使用中心差计算 x 和 y 方向的梯度
    dx = (image[:, :-2] - image[:, 2:]) / 2
    dy = (image[2:, :] - image[:-2, :]) / 2

    # # 填充渐变数组以匹配原始图像形状
    dx = np.pad(dx, ((0, 0), (1, 1)), mode='constant')
    dy = np.pad(dy, ((1, 1), (0, 0)), mode='constant')

    # 将梯度合并到一个数组中
    gradients = np.stack((dx, dy), axis=2)

    return gradients

def smooth_histogram(histogram):
    # 使用圆形邻域平滑直方图
    smoothed_histogram = np.zeros_like(histogram)

    for i in range(histogram.shape[0]):
        smoothed_histogram[i] = (histogram[i-1] + histogram[i] + histogram[(i+1) % histogram.shape[0]]) / 3

    return smoothed_histogram


def create_descriptor(histogram):
    descriptor = []
    # 连接子直方图以形成描述符
    for i in range(4):
        for j in range(4):
            sub_histogram = histogram[i: (i + 1), j: (j + 1), :]
            sub_descriptor = sub_histogram.flatten()
            descriptor.extend(sub_descriptor)
    descriptor = np.array(descriptor)

    # 对描述符设置阈值以抑制大值
    descriptor[descriptor > 0.2] = 0.2

    # 将描述符标准化为单位长度
    descriptor /= np.linalg.norm(descriptor)


    return descriptor


def match_keypoints(descriptors1, descriptors2, threshold=0.7):
    matches = []

    # 迭代第一张图像的描述符
    for i, desc1 in enumerate(descriptors1):
        best_match_index = -1
        best_match_distance = float('inf')

        # C将第一幅图像的描述符与第二幅图像的描述进行比较
        for j, desc2 in enumerate(descriptors2):
            # 计算描述符之间的欧几里德距离
            distance = np.linalg.norm(desc1 - desc2)

            # 如果距离小于目前为止的最佳匹配，则更新最佳匹配
            if distance < best_match_distance:
                best_match_index = j
                best_match_distance = distance

        # 仅考虑距离低于特定阈值的匹配
        if best_match_distance < threshold:
            match = cv2.DMatch(i, best_match_index, best_match_distance)
            matches.append(match)
            print(match.queryIdx, match.trainIdx)
    return matches


def draw_keypoints(image, keypoints):
    image_with_keypoints = image.copy()

    for kp in keypoints:
        x, y = kp[:2]  # 只获取前两个值

        # 绘制关键点
        cv2.circle(image_with_keypoints, (int(x), int(y)), 2, (255, 100, 100), -1)

    return image_with_keypoints

def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    # 计算拼接后图像的高度
    height = max(image1.shape[0], image2.shape[0])

    # 将图像调整为相同的高度
    image1_resized = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
    image2_resized = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

    # 拼接图像
    image_matches = np.concatenate((image1_resized, image2_resized), axis=1)

    # 绘制关键点和匹配线
    offset = image1_resized.shape[1]
    m = matches[:10]
    for match in m:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]) + offset, int(kp2[1]))
        cv2.circle(image_matches, pt1, 2, (100, 255, 100), 1)
        cv2.circle(image_matches, pt2, 2, (0, 255, 0), 1)
        cv2.line(image_matches, pt1, pt2, (10, 150, 255), 1)

    return image_matches

# 加载输入图像
image1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)

# 构建高斯金字塔
n = 2
num_octaves = int(np.log2(min(image1.shape[0], image1.shape[1])) - 3)
num_scales = n + 3
sigma = 1.6
T = 0.04
God1, pyramid1 = build_gaussian_pyramid(image1, num_octaves, num_scales, sigma)
God2, pyramid2 = build_gaussian_pyramid(image2, num_octaves, num_scales, sigma)

# 执行关键点检测
keypoints1 = detect_keypoints(God1)
keypoints2 = detect_keypoints(God2)

# 计算描述符
descriptors1 = compute_descriptors(image1, keypoints1)
descriptors2 = compute_descriptors(image2, keypoints2)

# 匹配关键点
matches = match_keypoints(descriptors1, descriptors2)

# 绘制关键点和匹配
image1_with_keypoints = draw_keypoints(image1, keypoints1)
image2_with_keypoints = draw_keypoints(image2, keypoints2)
image_matches = draw_matches(image1, image2, keypoints1, keypoints2, matches)

# 显示图像
# plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image1_with_keypoints, cmap='gray'), plt.title('Image 1 with Keypoints')
plt.subplot(122), plt.imshow(image2_with_keypoints, cmap='gray'), plt.title('Image 2 with Keypoints')
plt.show()
plt.imshow(image_matches, cmap='gray'), plt.title('Matches')
plt.show()