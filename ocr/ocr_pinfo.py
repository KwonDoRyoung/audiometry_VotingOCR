# -*-coding: utf-8
import logging
import cv2 as cv
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt

from .vision import *
from .utils import extract_and_recognize_characters
from .utils import count_characters_in_image, calculate_string_width, calculate_string_similarity
from .utils import crop_and_trim, find_table_grid_points, process_image

logger = logging.getLogger(__name__)


def find_row(src):
    _inv_src = 255 - src
    _h = src.shape[0]
    _h_dist = np.sum(_inv_src, axis=1)
    row_points = []
    in_non_zero_region = False
    start = 0
    for idx in range(_h):
        if _h_dist[idx] != 0 and not in_non_zero_region:
            in_non_zero_region = True
            start = idx
        elif _h_dist[idx] ==0 and in_non_zero_region:
            in_non_zero_region = False
            row_points.append((start, idx-1))

    if in_non_zero_region:
        row_points.append((start, _h - 1))

    return row_points

def find_column(src):
    _inv_src = 255 - src
    _w = src.shape[1]
    _w_dist = np.sum(_inv_src, axis=0)
    _col_points = []
    in_non_zero_region = False
    start = 0
    for idx in range(_w):
        if _w_dist[idx] != 0 and not in_non_zero_region:
            in_non_zero_region = True
            start = idx
        elif _w_dist[idx] ==0 and in_non_zero_region:
            in_non_zero_region = False
            _col_points.append((start, idx-1))

    if in_non_zero_region:
        _col_points.append((start, _w - 1))

    return _col_points


def ocr_age_region(src, h1, h2, padding, display=False):
    ocr_cfg = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'

    _dst = crop_by_coordinate(src, h1 - padding, h2 + padding, 750, src.shape[1], display=False)
    col_points = find_column(_dst)
    _age_points = []
    for w1, w2 in col_points:
        if w1 < 225:
            continue
        _age_points.append(w1)
        _age_points.append(w2)
    
    if len(_age_points) >= 2:
        _age_region = _dst.copy()[:, _age_points[0]: _age_points[-1]]
        _age_region = add_padding(_age_region, padding=10, display=False)
        _age_region = cv.morphologyEx(_age_region, op=cv.MORPH_DILATE, kernel=np.ones([3,3]))
        _age_region = cv.morphologyEx(_age_region, op=cv.MORPH_ERODE, kernel=np.ones([2,2]))

        _data = pytesseract.image_to_string(_age_region, config=ocr_cfg).strip()
        if display:
            fig, axs = plt.subplots()
            axs.imshow(_age_region, cmap="gray")
            axs.set_title(f"{_data}")
            plt.show()
        return _data
    else:
        return "-"
    
def ocr_date_region01(src, h1, h2, padding, display=False):
    ocr_cfg = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789-'

    _dst = crop_by_coordinate(src, h1 - padding, h2 + padding, 500, src.shape[1], display=False)
    _dst = add_padding(_dst, padding=5, display=False)
    if np.sum(255-_dst) < 1000:
        return "-"
    else:        
        _data = pytesseract.image_to_string(_dst, config=ocr_cfg).strip()

        if display:
            fig, axs = plt.subplots()
            axs.imshow(_dst, cmap="gray")
            axs.set_title(f"{_data}")
            plt.show()
        return _data

def ocr_date_region02(src, h1, h2, padding, display=False):
    ocr_cfg = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789-'
    _dst = crop_by_coordinate(src, h1 - padding, h2 + padding, 300, src.shape[1]-290, display=False)
    _data = pytesseract.image_to_string(_dst, config=ocr_cfg).strip()
    _data_list = _data.split("-")
    _data_list = [item for item in _data_list if item != '']
    if len(_data_list) == 0:
        return "-"
    elif len(_data_list) == 1:
        _data = _data_list[0]
    elif len(_data_list) == 2:
        _data = _data_list[-1]
    elif len(_data_list) > 2:
        _data = _data_list[1]
        
    if display:
        fig, axs = plt.subplots()
        axs.imshow(_dst, cmap="gray")
        axs.set_title(f"{_data}")
        plt.show()
    return _data


def run_ocr_PInfo(src, y1:int = 0, y2:int = 160, x1:int = 350, x2:int = 800, pad:int=10, display=False):
    target_image = None
    try:
        target_image = crop_by_coordinate(src, y1, y2, x1, x2)
    except Exception as e:
        logger.exception("Error in crop_by_coordinate in run_ocr_PInfo")
        return None

    if target_image is None:
        logger.error("crop_by_coordinate returned None, skipping crop_and_trim. in run_ocr_PInfo")
        return None
    
    try:
        target_image = crop_and_trim(target_image, padding=pad)
    except Exception as e:
        logger.exception(f"crop_and_trim; {e}. in run_ocr_PInfo")
        return None

    if target_image is None:
        logger.error("target_image returned None, skipping run_ocr_PInfo.")
        return None
    
    try:
        scale_factor = 4
        target_image = resize_image(target_image, scale_factor_x=scale_factor, scale_factor_y=scale_factor)
        target_image = threshold(target_image, thresh_type=cv.THRESH_BINARY + cv.THRESH_OTSU, blur_size=[7,7], display=display)

        row_points = find_row(target_image)

        padding = 10
        h1, h2 = row_points[2]
        age_data = ocr_age_region(target_image, h1, h2, padding, display=display)
        
        h1, h2 = row_points[4]
        date_data01 = ocr_date_region01(target_image, h1, h2, padding, display=display)

        h1, h2 = row_points[5]
        date_data02 = ocr_date_region02(target_image, h1, h2, padding, display=display)

        _pinfo_data = {"Age": age_data, "date01": date_data01, "date02": date_data02}
    except:
        logger.error("target_image returned None, skipping run_ocr_PInfo.")
        return None
    
    return _pinfo_data