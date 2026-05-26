# -*-coding: utf-8
from typing import Optional, List
import logging
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ..vision import threshold

logger = logging.getLogger(__name__)

def crop_and_trim(src, padding, display=False):
    if not isinstance(src, np.ndarray):
        logger.error(f"Expected src to be np.ndarray, but got {type(src)}.")
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)}.")

    h, w = src.shape[:2]
       
    th_src = threshold(src, 
                       thresh_type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU, 
                       blur_size=[7,7]) 

    w_dist = np.sum(th_src, axis=0)
    h_dist = np.sum(th_src, axis=1)

    l_idx = 0
    for l_idx in range(w):
        if w_dist[l_idx] > 0:
            l_idx = max(l_idx - padding, 0)
            break

    r_idx = w - 1
    for r_idx in range(w-1, 0, -1):
        if w_dist[r_idx] > 0:
            r_idx = min(r_idx + padding, w)
            break

    t_idx = 0
    for t_idx in range(h):
        if h_dist[t_idx] > 0:
            t_idx = max(t_idx - padding, 0)
            break

    b_idx = h - 1
    for b_idx in range(h-1, 0, -1):
        if h_dist[b_idx] > 0:
            b_idx = min(b_idx + padding, h)
            break

    if len(src.shape) == 3:
        _dst = src[t_idx:b_idx, l_idx:r_idx, :].copy()
    elif len(src.shape) == 2:
        _dst = src[t_idx:b_idx, l_idx:r_idx].copy()
    else:
        logger.error(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")
        raise RuntimeError(f"Unsupported image shape: {src.shape}. Only 2D or 3D images are supported.")

    if display:
        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Crop and Trim Results")
        
        axs[0, 0].imshow(th_src, cmap="gray")
        axs[0, 0].set_title("Thresholded Image")
        
        axs[0, 1].plot(h_dist, range(len(h_dist)))
        axs[0, 1].set_ylim(h, 0)
        axs[0, 1].set_title("Vertical Sum")
        
        axs[1, 0].plot(w_dist)
        axs[1, 0].set_xlim(0, w)
        axs[1, 0].set_title("Horizontal Sum")
        
        axs[1, 1].imshow(_dst, cmap="gray")
        axs[1, 1].set_title("Cropped Image")
        
        plt.tight_layout()
        plt.show()
    
    return _dst


def find_table_grid_points(src: np.ndarray, n_peak_x: int, n_peak_y: int, 
                           threshold_w: int, threshold_h: int, 
                           canny_threshold1: int = 50, canny_threshold2: int = 200,
                           display: bool = False) -> Optional[List[List[int]]]:
    if not isinstance(src, np.ndarray):
        logging.error(f"IndexError when creating bounding box: {e}")
        raise IndexError(f"IndexError when creating bounding box: {e}")
    
    if len(src.shape) == 3:
        src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    _dst = cv.Canny(src, canny_threshold1, canny_threshold2, None, 3)
    
    _w_dist = np.sum(_dst, axis=0)
    _h_dist = np.sum(_dst, axis=1)
    
    _w_peak, _ = find_peaks(_w_dist, height=threshold_w)
    _h_peak, _ = find_peaks(_h_dist, height=threshold_h)

    if len(_w_peak) != n_peak_x or len(_h_peak) != n_peak_y:
        logging.error(f"Expected {n_peak_x} x-peaks and {n_peak_y} y-peaks, but found {len(_w_peak)} and {len(_h_peak)}.")
        raise ValueError(f"Expected {n_peak_x} x-peaks and {n_peak_y} y-peaks, but found {len(_w_peak)} and {len(_h_peak)}.")
    
    _x_points = sorted(_w_peak.tolist())[1:-1]
    _y_points = sorted(_h_peak.tolist())[1:-1]

    bounding_box_list = []
    for y_idx in range(0,len(_y_points),2):
        row_bounding_box_list = []
        for x_idx in range(0,len(_x_points),2):
            try:
                x1, y1 = _x_points[x_idx], _y_points[y_idx]
                w = _x_points[x_idx + 1] - x1
                h = _y_points[y_idx + 1] - y1
                row_bounding_box_list.append((x1, y1, w, h))
            except IndexError as e:
                logging.error(f"IndexError when creating bounding box: {e}", exc_info=True)
                return
        bounding_box_list.append(row_bounding_box_list)

    if display:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f"src = {src.dtype, src.shape}, peak(w,h)={len(_w_peak), len(_h_peak)}")
        
        axs[0, 0].imshow(src, cmap="gray")
        axs[0, 0].set_title("Original Image")
        
        axs[0, 1].plot(_h_dist, range(len(_h_dist)))
        axs[0, 1].set_ylim(src.shape[0], 0)
        axs[0, 1].set_title("Horizontal Projection")
        
        axs[1, 0].plot(_w_dist)
        axs[1, 0].set_xlim(0, src.shape[1])
        axs[1, 0].set_title("Vertical Projection")
        
        axs[1, 1].imshow(_dst, cmap="gray")
        axs[1, 1].set_title("Canny Edges")
        for row_bounding_box_list in bounding_box_list:
            for x, y, _, _ in row_bounding_box_list:
                axs[1, 1].add_patch(plt.Circle((x, y), radius=5, color='red', fill=False))
        
        plt.tight_layout()
        plt.show()

    return bounding_box_list
