#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import cv2
from skimage import exposure, morphology


class Pretreatment(object):

    def __init__(self, origin):
        self.origin = origin

    def get(self):
        return self.origin

    def binarization(self, value, modify=False):
        ret, _binarization = cv2.threshold(self.origin, value, 255, cv2.THRESH_BINARY)
        if modify:
            self.origin = _binarization
        return _binarization

    def median_blur(self, value, modify=False):
        if not value:
            return self.origin
        value = value + 1 if value % 2 == 0 else value
        _smooth = cv2.medianBlur(self.origin, value)
        if modify:
            self.origin = _smooth
        return _smooth

    def gaussian_blur(self, value, modify=False):
        if not value:
            return self.origin
        value = value + 1 if value % 2 == 0 else value
        _blur = cv2.GaussianBlur(self.origin, (value, value), 0)
        if modify:
            self.origin = _blur
        return _blur

    def noise_remove(self, value,modify=False):
        """
        8邻域降噪
        Args:
            image_name: 图片文件命名
            k: 判断阈值

        Returns:

        """

        def calculate_noise_count(img_obj, w, h):
            """
            计算邻域非白色的个数
            Args:
                img_obj: img obj
                w: width
                h: height
            Returns:
                count (int)
            """
            count = 0
            width, height = img_obj.shape
            for _w_ in [w - 1, w, w + 1]:
                for _h_ in [h - 1, h, h + 1]:
                    if _w_ > width - 1:
                        continue
                    if _h_ > height - 1:
                        continue
                    if _w_ == w and _h_ == h:
                        continue
                    if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
                        count += 1
            return count

        # 灰度
        gray_img = self.origin
        w, h = gray_img.shape
        for _w in range(w):
            for _h in range(h):
                if _w == 0 or _h == 0:
                    gray_img[_w, _h] = 255
                    continue
                # 计算邻域pixel值小于255的个数
                pixel = gray_img[_w, _h]
                if pixel == 255:
                    continue

                if calculate_noise_count(gray_img, _w, _h) < value:
                    gray_img[_w, _h] = 255
        if modify:
            self.origin = gray_img
        return gray_img

    def adjust_gamma(self, value, modify=False):
        if not value:
            return self.origin
        gray_img = exposure.adjust_gamma(self.origin, value)
        if modify:
            self.origin = gray_img
        return gray_img


def preprocessing(image, gamma=-1,binaryzation=-1, smooth=-1, blur=-1):
    pretreatment = Pretreatment(image)
    if gamma>0:
        pretreatment.adjust_gamma(gamma,True)
    if binaryzation > 0:
        pretreatment.binarization(binaryzation, True)
    if smooth != -1:
        pretreatment.median_blur(smooth, True)
    if blur != -1:
        pretreatment.gaussian_blur(blur, True)
    # cv2.imshow('img', pretreatment.get())
    # cv2.waitKey(1000)
    return pretreatment.get()


if __name__ == '__main__':
    pass
