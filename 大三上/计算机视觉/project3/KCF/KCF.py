import cv2
import numpy as np
from skimage.feature import hog


class KCFTracker:
    def __init__(self):
        self.image = None
        self.pos = None
        self.target_size = None
        self.sigma = 0.5
        self.lmbda = 0.01
        self.interp_factor = 0.02
        self.target = None
        self.alphaf = None

    def get_subwindow(self, image, pos, size):
        x, y = pos
        w, h = size
        x1 = int(round(x - w / 2))
        y1 = int(round(y - h / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + h / 2))
        patch = image[y1:y2, x1:x2]
        # # 提取图像块
        # patch = image[y1:y2, x1:x2]
        # if patch.size == 0:
        #     raise ValueError("提取的图像块为空!")
        #
        #
        # if patch.ndim == 3:
        #     patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        #
        # # 添加特征提取步骤
        # hog_features = hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

        return patch

    def gaussian_correlation(self, xf, yf):
        N = xf.shape[0] * xf.shape[1]
        xf_conj = np.conj(xf)
        xyf = xf * yf.conj()
        xyf_sum = np.sum(xyf, axis=2)
        x_sum = np.sum(xf, axis=2)
        y_sum = np.sum(yf, axis=2)

        numerator = xyf_sum - (x_sum * y_sum / N)
        denominator = np.sqrt((np.sum(np.abs(xf_conj) ** 2, axis=2) - (np.abs(x_sum) ** 2 / N)) *
                              (np.sum(np.abs(yf.conj()) ** 2, axis=2) - (np.abs(y_sum) ** 2 / N)) + 1e-16)
        response = np.real(np.fft.ifft2(numerator / denominator))
        return response

    def train(self, x, y):
        xf = np.fft.fft2(x, axes=(0, 1))
        yf = np.fft.fft2(y, axes=(0, 1))
        kf = self.gaussian_correlation(xf, yf)
        window = self.window_f(kf.shape)
        alphaf = np.fft.fft2(window * np.fft.ifft2(kf), axes=(0, 1)) / (np.fft.fft2(kf, axes=(0, 1)) + self.lmbda)
        alphaf = np.resize(alphaf, self.alphaf.shape)
        alphaf = (1 - self.interp_factor) * self.alphaf + self.interp_factor * alphaf

        return alphaf

    def initial(self, image, pos, size):
        self.image = image
        self.pos = pos
        self.target_size = size
        self.target = self.get_subwindow(image, self.pos, self.target_size)
        output_sigma = np.sqrt(np.prod(self.target_size)) * self.sigma / 8
        rs, cs = np.ogrid[0:self.target_size[0], 0:self.target_size[1]]
        y = np.exp(
            -0.5 * ((rs - self.target_size[0] / 2) ** 2 + (cs - self.target_size[1] / 2) ** 2) / output_sigma ** 2)
        yf = np.fft.fft2(y, axes=(0, 1)).T
        kf = self.gaussian_correlation(self.target, self.target)
        print(kf.shape)
        kf = np.resize(kf, size).T
        self.alphaf = yf / (np.fft.fft2(kf, axes=(0, 1)) + self.lmbda)

        # print('self.alphaf', self.alphaf)

    def update(self, image):
        self.image = image
        new_pos = self.detect(self.image)
        print(new_pos)
        if self.target is not None:  # 添加检查以确保self.target不为空
            tar = self.target
            self.target = self.get_subwindow(self.image, self.pos, self.target_size)
            self.alphaf = self.train(self.target, tar)
        return new_pos

    def detect(self, image):
        xf = np.fft.fft2(self.target, axes=(0, 1))
        zf = np.fft.fft2(self.get_subwindow(image, self.pos, self.target_size), axes=(0, 1))
        kzf = self.gaussian_correlation(zf, xf)
        kzf = np.resize(kzf, self.alphaf.shape)
        response = np.real(np.fft.ifft2(self.alphaf * kzf))
        max_response = np.unravel_index(np.argmax(response), response.shape)
        dy = max_response[0] - response.shape[0] // 2
        dx = max_response[1] - response.shape[1] // 2
        new_pos = (self.pos[0] + dx, self.pos[1] + dy)
        return new_pos

    # 添加一个窗口函数
    def window_f(self, size):

        hann1d_y = np.hanning(size[0])
        hann1d_x = np.hanning(size[1])
        # 确保窗口大小与 size 参数匹配
        hann2d = np.outer(np.hanning(size[0]), np.hanning(size[1]))
        return hann2d