# -*-coding: utf-8
from typing import Union, List, Tuple

import logging
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

def add_padding(src, padding: Union[int, List, Tuple],
                background_value: int = 255, display: bool = False):
    if isinstance(padding, int):
        padding = [padding, padding, padding, padding]
    elif len(padding) == 2:
        padding = [padding[0], padding[0], padding[1], padding[1]]
    elif len(padding) != 4:
        raise ValueError("Padding must be an int, a list of length 2, or a list of length 4.")

    if any(p < 0 for p in padding):
        raise ValueError("Padding values must be non-negative.")

    if not isinstance(src, np.ndarray):
        raise RuntimeError(f"Unsupported data type: {type(src)}. Expected np.ndarray.")
    
    if src.dtype not in [np.uint8, np.float32]:
        raise ValueError(f"Unsupported image dtype: {src.dtype}. Expected uint8 or float32.")

    if not (0 <= background_value <= 255):
        raise ValueError("background_value must be between 0 and 255.")

    if len(src.shape) == 3: 
        ori_h, ori_w, ori_ch = src.shape
    elif len(src.shape) == 2:
        ori_h, ori_w = src.shape
        ori_ch = 1
    else:
        raise RuntimeError(f"Unsupported image shape: {src.shape}. (shape: {src.shape})")

    # _dst 초기화: 배경 색상을 사용
    if ori_ch == 3:
        _dst = np.ones((ori_h + padding[0] + padding[1],
                        ori_w + padding[2] + padding[3],
                        ori_ch,), dtype=src.dtype) * background_value
    else:
        _dst = np.ones((ori_h + padding[0] + padding[1], 
                        ori_w + padding[2] + padding[3]), dtype=src.dtype) * background_value

    if ori_ch == 3:
        _dst[padding[0] : padding[0] + ori_h, padding[2] : padding[2] + ori_w, :] = src.copy()
    else:
        _dst[padding[0] : padding[0] + ori_h, padding[2] : padding[2] + ori_w] = src.copy()

    if display:
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(src, cmap="gray" if ori_ch == 1 else None)
        axs[0].set_title(f"Original Image {src.shape}")
        axs[1].imshow(_dst, cmap="gray" if ori_ch == 1 else None)
        axs[1].set_title(f"Padding Image {_dst.shape}")
        plt.tight_layout()
        plt.show()

    return _dst.astype(src.dtype)


def crop_by_coordinate(src, y1: int, y2: int, x1: int, x2: int,
                       y1_offset: int = 0, y2_offset: int = 0,
                       x1_offset: int = 0, x2_offset: int = 0,
                       display=False):
    _y1 = max(0, y1 + y1_offset)
    _y2 = min(src.shape[0], y2 + y2_offset)
    _x1 = max(0, x1 + x1_offset)
    _x2 = min(src.shape[1], x2 + x2_offset)

    if _y1 >= _y2 or _x1 >= _x2:
        logger.critical(f"Invalid cropping coordinates: y1 >= y2 or x1 >= x2. Got ({_y1}, {_y2}, {_x1}, {_x2})")
        raise ValueError(f"Invalid cropping coordinates: y1 >= y2 or x1 >= x2. Got ({_y1}, {_y2}, {_x1}, {_x2})")
        
    if not isinstance(src, np.ndarray):
        logger.error(f"Unsupported data type: {type(src)}. Expected np.ndarray.")
        raise RuntimeError(f"Unsupported data type: {type(src)}. Expected np.ndarray.")

    if len(src.shape) == 3:
        _dst = src[_y1:_y2, _x1:_x2, :]
    elif len(src.shape) == 2:
        _dst = src[_y1:_y2, _x1:_x2]
    else:
        logger.error(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")
        raise RuntimeError(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")

    if display:
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(src, cmap=None if len(src.shape) == 3 else "gray")
        axs[0].set_title(f"Original Image {src.shape}")
        axs[1].imshow(_dst, cmap=None if len(_dst.shape) == 3 else "gray")
        axs[1].set_title(f"Cropped Image {_dst.shape}")
        plt.tight_layout()
        plt.show()

    return _dst


def crop_by_bounding_box(src, x: int, y: int, w: int, h: int,
                         x_offset: int = 0, y_offset: int = 0,
                         w_margin: int = 0, h_margin: int = 0,
                         display=False):
    if not isinstance(src, np.ndarray):
        logger.error(f"Unsupported data type: {type(src)}. Expected np.ndarray.")
        raise RuntimeError(f"Unsupported data type: {type(src)}. Expected np.ndarray.")

    _x = max(0, x + x_offset)
    _y = max(0, y + y_offset)
    _w = min(src.shape[1] - _x, w + w_margin)
    _h = min(src.shape[0] - _y, h + h_margin)

    if _w <= 0 or _h <= 0:
        logger.critical(f"Invalid bounding box dimensions: width and height must be positive. Got width={_w}, height={_h}")
        raise ValueError(f"Invalid bounding box dimensions: width and height must be positive. Got width={_w}, height={_h}")

    if len(src.shape) == 3:
        _dst = src[_y : _y + _h, _x : _x + _w, :]
    elif len(src.shape) == 2:
        _dst = src[_y : _y + _h, _x : _x + _w]
    else:
        logger.error(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")
        raise RuntimeError(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")

    if display:
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(src, cmap=None if len(src.shape) == 3 else "gray")
        axs[0].set_title(f"Original Image {src.shape}")
        axs[1].imshow(_dst, cmap=None if len(_dst.shape) == 3 else "gray")
        axs[1].set_title(f"Cropped Image {_dst.shape}")
        plt.tight_layout()
        plt.show()

    return _dst
