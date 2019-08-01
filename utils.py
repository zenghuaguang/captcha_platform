#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import io
import re
import cv2
import time
import base64
import functools
import binascii
import datetime
import hashlib
import numpy as np
from PIL import Image as PIL_Image
from constants import Response, Config
from pretreatment import preprocessing
from config import ModelConfig


class Arithmetic(object):

    def calc(self, formula):
        formula = re.sub(' ', '', formula)
        formula_ret = 0
        match_brackets = re.search(r'\([^()]+\)', formula)
        if match_brackets:
            calc_result = self.calc(match_brackets.group().strip("(,)"))
            formula = formula.replace(match_brackets.group(), str(calc_result))
            return self.calc(formula)
        else:
            formula = formula.replace('--', '+').replace('++', '+').replace('-+', '-').replace('+-', '-')
            while re.findall(r"[*/]", formula):
                get_formula = re.search(r"[.\d]+[*/]+[-]?[.\d]+", formula)
                if get_formula:
                    get_formula_str = get_formula.group()
                    if get_formula_str.count("*"):
                        formula_list = get_formula_str.split("*")
                        ret = float(formula_list[0]) * float(formula_list[1])
                    else:
                        formula_list = get_formula_str.split("/")
                        ret = float(formula_list[0]) / float(formula_list[1])
                    formula = formula.replace(get_formula_str, str(ret)).replace('--', '+').replace('++', '+')
            formula = re.findall(r'[-]?[.\d]+', formula)
            for num in formula:
                formula_ret += float(num)
        return formula_ret


class ParamUtils(object):

    @staticmethod
    def filter(param):
        if isinstance(param, list) and len(param) > 0 and isinstance(param[0], bytes):
            return param[0].decode()
        return param


class SignUtils(object):

    @staticmethod
    def md5(text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @staticmethod
    def timestamp():
        return int(time.mktime(datetime.datetime.now().timetuple()))


class PathUtils(object):

    @staticmethod
    def get_file_name(path: str):
        if '/' in path:
            return path.split('/')[-1]
        elif '\\' in path:
            return path.split('\\')[-1]
        else:
            return path


class ImageUtils(object):

    def __init__(self, model: ModelConfig):
        self.model = model

    @staticmethod
    def get_bytes_batch(base64_or_bytes):
        response = Response()
        try:
            if isinstance(base64_or_bytes, bytes):
                bytes_batch = [base64_or_bytes]
            elif isinstance(base64_or_bytes, list):
                bytes_batch = [base64.b64decode(i.encode('utf-8')) for i in base64_or_bytes if isinstance(i, str)]
                if not bytes_batch:
                    bytes_batch = [base64.b64decode(i) for i in base64_or_bytes if isinstance(i, bytes)]
            else:
                bytes_batch = base64.b64decode(base64_or_bytes.encode('utf-8')).split(Config.split_flag)
        except binascii.Error:
            return None, response.INVALID_BASE64_STRING
        what_img = [ImageUtils.test_image(i) for i in bytes_batch]
        if None in what_img:
            return None, response.INVALID_IMAGE_FORMAT
        return bytes_batch, response.SUCCESS

    @staticmethod
    def get_image_batch(model: ModelConfig, bytes_batch):
        # Note that there are two return objects here.
        # 1.image_batch, 2.response

        response = Response()

        def load_image(image_bytes):
            if isinstance(image_bytes, bytes):
                nparr = np.fromstring(image_bytes, np.uint8)
                im = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            else:
                im = cv2.imread(image_bytes, cv2.IMREAD_GRAYSCALE)
            # The OpenCV cannot handle gif format images, it will return None.
            if im is None:
                data_stream = io.BytesIO(image_bytes)
                pil_image = PIL_Image.open(data_stream)
                rgb = pil_image.split()
                size = pil_image.size

                if len(rgb) > 3 and model.replace_transparent:
                    background = PIL_Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, (0, 0, size[0], size[1]), pil_image)
                    pil_image = background

                if model.image_channel == 1:
                    pil_image = pil_image.convert('L')
                im = np.array(pil_image)
            image = preprocessing(im,model.gamma ,model.binaryzation, model.smooth, model.blur).astype(
                np.float32)
            if model.resize[0] == -1:
                ratio = model.resize[1] / size[1]
                resize_width = int(ratio * size[0])
                image = cv2.resize(image, (resize_width, model.resize[1]))
            else:
                image = cv2.resize(image, (model.resize[0], model.resize[1]))
            image = image.swapaxes(0, 1)
            return (image[:, :, np.newaxis] if model.image_channel == 1 else image[:, :]) / 255.

        try:
            image_batch = [load_image(i) for i in bytes_batch]
            return image_batch, response.SUCCESS
        except OSError:
            return None, response.IMAGE_DAMAGE
        except ValueError as _e:
            print(_e)
            return None, response.IMAGE_SIZE_NOT_MATCH_GRAPH

    @staticmethod
    def pil_image(image_bytes):
        data_stream = io.BytesIO(image_bytes)
        pil_image = PIL_Image.open(data_stream).convert('RGB')
        return pil_image

    @staticmethod
    def size_of_image(image_bytes: bytes):
        _null_size = tuple((-1, -1))
        try:
            data_stream = io.BytesIO(image_bytes)
            size = PIL_Image.open(data_stream).size
            return size
        except OSError:
            return _null_size
        except ValueError:
            return _null_size

    @staticmethod
    def test_image(h):
        """JPEG"""
        if h[:3] == b"\xff\xd8\xff":
            return 'jpeg'
        """PNG"""
        if h[:8] == b"\211PNG\r\n\032\n":
            return 'png'
        """GIF ('87 and '89 variants)"""
        if h[:6] in (b'GIF87a', b'GIF89a'):
            return 'gif'
        """TIFF (can be in Motorola or Intel byte order)"""
        if h[:2] in (b'MM', b'II'):
            return 'tiff'
        if h[:2] == b'BM':
            return 'bmp'
        """SGI image library"""
        if h[:2] == b'\001\332':
            return 'rgb'
        """PBM (portable bitmap)"""
        if len(h) >= 3 and \
                h[0] == b'P' and h[1] in b'14' and h[2] in b' \t\n\r':
            return 'pbm'
        """PGM (portable graymap)"""
        if len(h) >= 3 and \
                h[0] == b'P' and h[1] in b'25' and h[2] in b' \t\n\r':
            return 'pgm'
        """PPM (portable pixmap)"""
        if len(h) >= 3 and h[0] == b'P' and h[1] in b'36' and h[2] in b' \t\n\r':
            return 'ppm'
        """Sun raster file"""
        if h[:4] == b'\x59\xA6\x6A\x95':
            return 'rast'
        """X bitmap (X10 or X11)"""
        s = b'#define '
        if h[:len(s)] == s:
            return 'xbm'
        return None
