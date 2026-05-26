# -*-coding: utf-8
from typing import Optional, Union, List, Tuple

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def threshold(src: np.ndarray, thresh_type: int, 
              blur_size: Optional[Union[List[int], Tuple[int], int]] = None,
              display=False) -> np.ndarray:
    if not isinstance(src, np.ndarray):
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)} instead.")

    if blur_size is not None:
        if isinstance(blur_size, int):
            if blur_size <= 0:
                raise ValueError("blur_size must be a positive integer.")
            blur_size = [blur_size, blur_size]
        elif isinstance(blur_size, (list, tuple)):
            if len(blur_size) != 2 or any(b <= 0 for b in blur_size):
                raise ValueError("blur_size must contain two positive integers.")
        else:
            raise TypeError("blur_size must be an int, list, or tuple.")
    
    if blur_size is not None:
        src = cv.GaussianBlur(src, blur_size, 0)

    _, _dst = cv.threshold(src, 0, 255, thresh_type)

    if display:
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(src, cmap = None if len(src.shape) == 3 else "gray")
        axs[0].set_title(f"Original Image")
        axs[1].imshow(_dst, cmap = "gray")
        axs[1].set_title(f"Thresholded Image")
        plt.tight_layout()
        plt.show()

    return _dst.astype(src.dtype)


def resize_image(src: np.ndarray, scale_factor_x: float, scale_factor_y: float, display=False) -> np.ndarray:
    if not isinstance(src, np.ndarray):
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)} instead.")
    
    if scale_factor_x <= 0 or scale_factor_y <= 0:
        raise ValueError("Scale factors must be positive numbers.")

    _h, _w = src.shape[:2]
    
    _dsize = (int(_w * scale_factor_x), int(_h * scale_factor_y))
    
    _dst = cv.resize(src, dsize=_dsize)

    if display:
        _, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(src, cmap=None if len(src.shape) == 3 else "gray")
        axs[0].set_title(f"Original Image {src.shape}")
        axs[1].imshow(_dst, cmap=None if len(_dst.shape) == 3 else "gray")
        axs[1].set_title(f"Resized Image {_dst.shape}")
        plt.tight_layout()
        plt.show()

    return _dst.astype(src.dtype)
