# -*-coding: utf-8
import socket
import logging
import pytesseract

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ..vision import *

logger = logging.getLogger(__name__)


def load_workspace_path():
    workspace_paths = {
        'abrlab': "/home/abrlab_doyoung/",
        'LDS12': "/raid/doho/",
        'LDS13': "/raid/doyoung/",
        'AIMLMLSRV03': "/home/doyoung/",
    }
    hostname = socket.gethostname()

    workspace_path = workspace_paths.get(hostname, False)
    if not workspace_path:
        raise ValueError(f"{hostname} is not set the workspace_path")
    else:
        return workspace_path

def tesseract_ocr(src:np.ndarray, config):
    _dst = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
    _str = pytesseract.image_to_string(_dst, config=config).strip()
    return _str


def process_image(src, x, y, w, h, op, kernel, padding, crop_cfg):
    _dst = crop_by_bounding_box(src, x, y, w, h, **crop_cfg)
    _dst = cv.morphologyEx(_dst, op=op, kernel=kernel)
    _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)
    _dst = cv.cvtColor(_dst, cv.COLOR_GRAY2RGB)
    if padding is not None:
        _dst = add_padding(_dst, padding=padding)
    return _dst

def extract_and_recognize_characters(src, bbox_str, config_dict):
    if len(src.shape) == 2:
        src = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
    result_str = ""
    for i, (_x, _y, _w, _h) in enumerate(bbox_str, start=1):
        _dst = crop_by_bounding_box(src, _x, _y, _w, _h)
        _dst = cv.cvtColor(_dst, cv.COLOR_RGB2GRAY)
        _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)
        _dst = cv.cvtColor(_dst, cv.COLOR_GRAY2RGB)
        _dst = add_padding(_dst, [20, 20])
        _char = pytesseract.image_to_string(_dst, config=config_dict[f"{i}th"]).strip()
        if _char == "":
            _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel=np.ones([2,2]))
            _char = pytesseract.image_to_string(_dst, config=config_dict[f"{i}th"]).strip()
        result_str = _char + result_str
    return result_str


# def check_single_numeric(src, padding, display=False):
#     print("   > Run check_single_numeric")
#     if isinstance(src, np.ndarray):
#         if len(src.shape) == 3:
#             src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#         elif len(src.shape) == 2:
#             pass
#         else:
#             raise RuntimeError(f"src is not supported. ({len(src.shape)})")
#     else:
#         raise RuntimeError("src is not supported.")
    
#     _dst = vision.threshold(src, type=cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#     w_dist = np.sum(_dst, axis=0)
#     w = src.shape[1]
#     for l_idx1 in range(w):
#         if w_dist[l_idx1] > 0:
#             break
#     for r_idx2 in range(w - 1,0,-1):
#         if w_dist[r_idx2] > 0:
#             break

#     for r_idx1 in range(l_idx1, w):
#         if w_dist[r_idx1] == 0:
#             break
#     for l_idx2 in range(r_idx2 - 1,0,-1):
#         if w_dist[l_idx2] == 0:
#             break
    
#     if r_idx2 - l_idx1 > 70: # Three number:
#         _dst01 = src.copy()[:, l_idx1 - padding//2:r_idx2 + padding//2]
#         _dst01 = vision.pad_image(_dst01, [10, 20], False)
#         return _dst01

#     _dst01 = src.copy()[:, l_idx1 - padding//2:r_idx1 + padding//2]
#     _dst01 = vision.pad_image(_dst01, [10, 20], False)

#     if r_idx1 > l_idx2: # SINGLE NUMBER
#         if display:
#             print(f"padding ( {padding} ): {l_idx1, r_idx1} -> {l_idx1 - padding, r_idx1 + padding}")
#             fig, axs = plt.subplots(1,2)
#             axs[0].plot(w_dist)
#             axs[1].imshow(_dst01, cmap="gray")
#             plt.show()
#         return [_dst01]
#     else:  # TWO NUMBER
#         _dst02 = src.copy()[:, l_idx2 - padding//2:r_idx2]
#         _dst02 = vision.pad_image(_dst02, [10, 20], False)

#         if display:
#             print(f"padding ( {padding} ): {l_idx1, r_idx1} -> {l_idx1 - padding, r_idx1 + padding}")
#             fig, axs = plt.subplots(2,2, figsize=(4, 4))
#             axs[0,0].plot(w_dist)
#             axs[1,0].imshow(_dst01, cmap="gray")
#             axs[0,1].imshow(_dst02, cmap="gray")
#             plt.show()
        
#         return [_dst01, _dst02]
    

def remove_outlier_region(src, padding, display=False):
    h = src.shape[0]
    w = src.shape[1]
       
    th_src = threshold(src, type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU) 
    w_dist = np.sum(th_src, axis=0)
    for l_idx in range(w):
        if w_dist[l_idx] > 0:
            l_idx = l_idx - padding
            break

    for r_idx in range(w-1,0,-1):
        if w_dist[r_idx] > 0:
            r_idx = r_idx + padding
            break
    
    h_dist = np.sum(th_src, axis=1)
    for t_idx in range(h):
        if h_dist[t_idx] > 0:
            t_idx = t_idx - padding
            break

    for b_idx in range(h-1,0,-1):
        if h_dist[b_idx] > 0:
            b_idx = b_idx + padding
            break

    _dst = src.copy()
    if isinstance(src, np.ndarray):
        if len(src.shape) == 3:
            _dst = src.copy()[t_idx:b_idx, l_idx:r_idx, :]
        elif len(src.shape) == 2:
            _dst = src.copy()[t_idx:b_idx, l_idx:r_idx]
        else:
            raise RuntimeError(f"{len(src.shape)} is not supported. src.shape={src.shape}")
    else:
        raise RuntimeError(f"{type(src)} is not supported.")

    if display:
        _, axs = plt.subplots(2,2)
        axs[0,0].imshow(th_src, cmap="gray")
        axs[0,1].plot(h_dist, range(len(h_dist)))
        axs[0,1].set_ylim(h, 0)
        axs[1,0].plot(w_dist)
        axs[1,1].imshow(_dst, cmap="gray")
        plt.tight_layout()
        plt.show()
    
    return _dst