# -*-coding: utf-8
import logging
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

from .vision import *
from .utils import crop_and_trim, find_table_grid_points

from .utils.string import is_blank_image, get_characters_width, calculate_string_similarity
from .utils.utils import tesseract_ocr

logger = logging.getLogger(__name__)

def get_string_width(src):
    width_dist = np.sum(cv.bitwise_not(src), axis=0)

    non_zero_indices = np.nonzero(width_dist)[0]

    if non_zero_indices.size == 0:
        # 양수 값이 없는 경우 (즉, 이미지가 비어 있는 경우)
        return 0

    # l_idx는 첫 번째 양수 인덱스, r_idx는 마지막 양수 인덱스입니다.
    l_idx = non_zero_indices[0]
    r_idx = non_zero_indices[-1]

    return r_idx - l_idx

def check_special_char(src):
    width = get_string_width(src)
    height_dist = np.sum(cv.bitwise_not(src), axis=1)

    zero_regions = np.where(height_dist == 0)[0]

    # 0인 구간이 있는지 확인
    if len(zero_regions) > 1:
        zero_gaps = np.split(zero_regions, np.where(np.diff(zero_regions) > 1)[0] + 1)
        middle_zero_exists = any(len(gap) > 1 and gap[0] > 0 and gap[-1] < len(height_dist) - 1 for gap in zero_gaps)
    else:
        middle_zero_exists = False

    # 봉우리 찾기 (0으로 나뉜 구간이 있는 경우만 봉우리로 인정)
    if len(zero_regions) > 1:
        peaks, _ = find_peaks(height_dist)
    else:
        peaks = []

    if middle_zero_exists and len(peaks) == 3 and width > 40:
        return "무"
    else:
        return None


def has_zero_in_middle(arr):
    return np.any(arr[1:-1] == 0)

def check_zero_five(src):
    # 0, 5 체크하는 알고리즘
    if len(src.shape) == 3:
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    
    w = src.shape[1]
    dst = 255. - src[:, :w//2-5]

    h_dist = np.sum(dst, axis=1)

    top_idx = 0
    for idx in range(len(h_dist)):
        if h_dist[idx] > 0:
            top_idx = idx
            break

    bottom_idx = 500
    for idx in range(len(h_dist)-1, 0, -1):
        if h_dist[idx] > 0:
            bottom_idx = idx
            break

    _has_zero = has_zero_in_middle(h_dist[top_idx:bottom_idx])
    if _has_zero:
        return "5"
    else:
        return "0"

def count_characters_and_widths(src, reverse=False):
    kernel = np.ones((2, 2), np.uint8)
    _dst = cv.erode(cv.bitwise_not(src), kernel, iterations=1)
    _dst = cv.dilate(_dst, kernel, iterations=1)

    contours, _ = cv.findContours(_dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    num_characters = len(contours)

    bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]
    bounding_boxes.sort(key=lambda box: box[0], reverse=reverse)

    # Merge boxes if they are close enough (x difference <= 10)
    merged_boxes = []
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            last_box = merged_boxes[-1]
            # Check if the current box is close enough to the last box
            if box[0] <= last_box[0] + 10:
                # Merge the boxes: calculate new coordinates and size
                x = min(last_box[0], box[0])
                y = min(last_box[1], box[1])
                w = max(last_box[0] + last_box[2], box[0] + box[2]) - x
                h = max(last_box[1] + last_box[3], box[1] + box[3]) - y
                merged_boxes[-1] = (x, y, w, h)
            else:
                merged_boxes.append(box)

    # Extract character widths from merged bounding boxes
    character_widths = [box[2] for box in merged_boxes]
    num_characters = len(merged_boxes)

    return num_characters, character_widths, merged_boxes



class SDT:
    def __init__(self, image: np.ndarray, is_verification:bool, y1:int = 710, y2:int = 870, x1:int = 350, x2:int = 570, padding:int=5, scale_factor:float=4):
        self.is_verification = is_verification
        self.region_coordinate = [y1, y2, x1, x2]
        self.region_image, self.cell_bounding_box_list = self.crop_region(image, y1, y2, x1, x2, padding, scale_factor)
        self.side_images = []
        self.percentile_images = []
        self.stimulus_images = []
    
    def crop_region(self, image, y1, y2, x1, x2, padding, scale_factor):
        _temp = crop_by_coordinate(image, y1, y2, x1, x2)
        _temp = crop_and_trim(_temp, padding=padding)

        _temp = resize_image(_temp, scale_factor_x=scale_factor, scale_factor_y=scale_factor)
        _temp = threshold(_temp, thresh_type=cv.THRESH_BINARY + cv.THRESH_OTSU)        
        # 원본: threshold_w=28000, threshold_h = 40000
        # 4배율: threshold_w=28000*4 = 112000, threshold_h = 40000*4 = 160000
        _cell_bounding_box_list = find_table_grid_points(_temp, 
                                                         n_peak_x=5*2, n_peak_y=8*2,
                                                         threshold_w=28000*scale_factor, 
                                                         threshold_h = 40000*scale_factor)
        
        return _temp, _cell_bounding_box_list
    
    def _ocr_side(self, x, y_points, w, h):
        _str_list = []
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image.copy(), x, y, w, h, 
                                            x_offset=5, y_offset=5, w_margin=-100, h_margin=-20)
                _dst = cv.morphologyEx(_dst, op=cv.MORPH_OPEN, kernel=np.ones([5, 5]))
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
                _dst = add_padding(_dst, padding=[10, 0])
                self.side_images.append({"image": _dst})

                _str = tesseract_ocr(_dst, config='--psm 10 --oem 3 -c tessedit_char_whitelist=RL')
                if _str not in ["R", "L"]:
                    _str = ""

                _str_list.append(_str)
            except Exception as e:
                logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                raise

        if "R" not in set(_str_list):
            _str_list = ["L" if _str == "L" else "R" for _str in _str_list]
        elif "L" not in set(_str_list):
            _str_list = ["R" if _str == "R" else "L" for _str in _str_list]

        if "R" in set(_str_list) and "L" in set(_str_list) and "" in set(_str_list):
            count_r = _str_list.count("R")
            count_l = _str_list.count("L")
            if count_r == 1:
                _str_list = ["R" if _str == "" else _str for _str in _str_list]
            if count_l == 1:
                _str_list = ["L" if _str == "" else _str for _str in _str_list]
        
        if any(_str not in ["R", "L"] for _str in _str_list):
            logger.error(f"Unexpected characters in side: {_str_list}")
            raise ValueError(f"Invalid characters in side: {_str_list}")
        
        for idx, _str in enumerate(_str_list):
            self.side_images[idx]["str"] = _str

        return _str_list

    def _ocr_percentile(self, x, y_points, w, h):
        _str_list = [] 
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image.copy(), x, y, w, h, x_offset=4, y_offset=4, w_margin=-4, h_margin=-4)
                _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel=np.ones([2, 2]))
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])

                _str = tesseract_ocr(_dst, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789%')
                if "%" in _str:
                    _str = _str.replace("%", "")
                _str_list.append(_str)
                self.percentile_images.append({"image": _dst, "str": _str})
            except Exception as e:
                logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                raise

        return _str_list

    def _ocr_percentile_verification(self, x, y_points, w, h, choice_string_operator, off_char_checker, off_remove_symbol):
        _str_list = [] 
        _percent_list = []
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image.copy(), x, y, w, h, x_offset=4, y_offset=4, w_margin=-4, h_margin=-4)
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)

                verification = VerificationSDTPercentile(cell_image=_dst, choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)
                # print(verification)
                _candidate_str_list = verification()
                # verification.display()
                if _candidate_str_list == "": # 공백
                    _main_str = ""
                    _main_voting_percent = 100
                else:
                    _main_str, _main_voting_percent = calculate_string_similarity(_candidate_str_list, num_char = verification.num_char)

                _str_list.append(_main_str)
                _percent_list.append(_main_voting_percent)
                self.percentile_images.append({"image": _dst, "str": _main_str, "%_candidate": _candidate_str_list})
            except Exception as e:
                logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                raise

        return _str_list, _percent_list

    def _ocr_stimulus(self, x, y_points, w, h):
        _str_list = [] 
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image.copy(), x, y, w, h,
                                            x_offset=10, y_offset=10,
                                            w_margin=-15, h_margin=-10)
                _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel=np.ones([2, 2]))
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
                _dst = add_padding(_dst, [30, 0])

                _str = tesseract_ocr(_dst, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789dB')
                if "d" in _str:
                    _str = _str.replace("d", "")
                if "B" in _str:
                    _str = _str.replace("B", "")

                _str_list.append(_str)
                self.stimulus_images.append({"image": _dst, "str": _str})
            except Exception as e:
                logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                raise

        return _str_list

    def _ocr_stimulus_verification(self, x, y_points, w, h, choice_string_operator, off_char_checker, off_remove_symbol):
        _str_list = [] 
        _percent_list = []
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image.copy(), x, y, w, h,
                                            x_offset=10, y_offset=10,
                                            w_margin=-15, h_margin=-10)
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)

                verification = VerificationSDTStimulus(cell_image=_dst, choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)
                # print(verification)
                _candidate_str_list = verification()
                # verification.display()
                if _candidate_str_list == "": # 공백
                    _main_str = ""
                    _main_voting_percent = 100
                else:
                    _main_str, _main_voting_percent = calculate_string_similarity(_candidate_str_list, num_char = verification.num_char)

                _str_list.append(_main_str)
                _percent_list.append(_main_voting_percent)
                self.stimulus_images.append({"image": _dst, "str": _main_str, "stimulus_candidate": _candidate_str_list})
            except Exception as e:
                logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                raise
            
        return _str_list, _percent_list

    def display(self):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.2, hspace=0.3)  # 3행 x 3열의 

        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(self.region_image, cmap='gray')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(self.side_images[0]["image"], cmap="gray")
        ax2.set_title(f'{self.side_images[0]["str"]}')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.imshow(self.side_images[1]["image"], cmap='gray')
        ax3.set_title(f'{self.side_images[1]["str"]}')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(self.percentile_images[0]["image"], cmap='gray')
        if self.is_verification:
            _candidate = ', '.join(self.percentile_images[0]["%_candidate"])
            ax4.set_title(f'{self.percentile_images[0]["str"]}\n{_candidate}')
        else:
            ax4.set_title(f'{self.percentile_images[0]["str"]}')
        ax4.axis('on')

        ax5 = fig.add_subplot(gs[2, 1])
        ax5.imshow(self.percentile_images[1]["image"], cmap='gray')
        if self.is_verification:
            _candidate = ', '.join(self.percentile_images[1]["%_candidate"])
            ax5.set_title(f'{self.percentile_images[1]["str"]}\n{_candidate}')
        else:
            ax5.set_title(f'{self.percentile_images[1]["str"]}')
        ax5.axis('on')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(self.stimulus_images[0]["image"], cmap='gray')
        if self.is_verification:
            _candidate = ', '.join(self.stimulus_images[0]["stimulus_candidate"])
            ax6.set_title(f'{self.stimulus_images[0]["str"]}\n{_candidate}')
        else:
            ax6.set_title(f'{self.stimulus_images[0]["str"]}')
        ax6.axis('on')

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.imshow(self.stimulus_images[1]["image"], cmap='gray')
        if self.is_verification:
            _candidate = ', '.join(self.stimulus_images[1]["stimulus_candidate"])
            ax7.set_title(f'{self.stimulus_images[1]["str"]}\n{_candidate}')
        else:
            ax7.set_title(f'{self.stimulus_images[1]["str"]}')
        ax7.axis('on')

        plt.tight_layout()
        plt.show()

    def __call__(self, choice_string_operator, off_char_checker, off_remove_symbol):
        x, y1, w, h = self.cell_bounding_box_list[1][0]
        _, y2, _, _ = self.cell_bounding_box_list[2][0]
        side_str = self._ocr_side(x, [y1, y2], w, h)
                
        x, _, _, _ = self.cell_bounding_box_list[0][1]
        if not self.is_verification:
            precentile_str = self._ocr_percentile(x, [y1, y2], w, h)
            percentile_voting_percent = None
        else:
            precentile_str, percentile_voting_percent = self._ocr_percentile_verification(x, [y1, y2], w, h, choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)
        
        x, _, _, _ = self.cell_bounding_box_list[0][2]
        if not self.is_verification:
            stimulus_str = self._ocr_stimulus(x, [y1, y2], w, h)
            stimulus_voting_percent = None
        else:
            stimulus_str, stimulus_voting_percent = self._ocr_stimulus_verification(x, [y1, y2], w, h, choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)
        
        table_data = {"Side": side_str, "%": precentile_str, "stimulus": stimulus_str, "voting_%": percentile_voting_percent, "voting_stimulus": stimulus_voting_percent}
        return pd.DataFrame(table_data)
        

class VerificationSDTPercentile:
    def __init__(self, cell_image:np.ndarray, choice_string_operator, off_char_checker, off_remove_symbol):
        self.cell_image = cell_image
        self.choice_string_operator = choice_string_operator
        self.off_char_checker = off_char_checker
        self.off_remove_symbol = off_remove_symbol

        self.rm_percentile_image = self.remove_percentile_str()
        self.is_blank = is_blank_image(self.rm_percentile_image)
    
        self.process_images = []
        self.string_width = 0
        self.num_char = 0
        self.char_images = []

        if not self.is_blank:
            self.process_images = self.get_various_images(self.rm_percentile_image.copy())
            self.string_width = get_string_width(self.rm_percentile_image.copy())
            self.num_char = self.get_num_char()
            if not self.off_char_checker:
                self.char_images = self.get_char_images(self.rm_percentile_image.copy())
    
    def get_num_char(self):
        if 0 < self.string_width <= 32:
            return 1
        elif 32 < self.string_width <= 62: # 두자리 숫자
            return 2
        elif 62 < self.string_width: # 세자리 숫자
            return 3

    def remove_percentile_str(self):
        if self.off_remove_symbol:
            return self.cell_image.copy()
        else:
            _dst = self.cell_image.copy()        
            _dst = cv.morphologyEx(_dst, op=cv.MORPH_ERODE, kernel=np.ones([5, 5]))
            _num_char, _widths, _x_points = get_characters_width(_dst)
            if len(_x_points) == 0:
                return np.zeros(self.cell_image.shape)
            else:
                x = _x_points[0]
                y = 0
                h = self.cell_image.shape[0]

            if _widths[-1] < 40: # ? % 떨어짐
                if _num_char == 1: # %
                    return np.zeros(self.cell_image.shape)
                else: # ? % / ? ? % / ? ? ? % 
                    w = _x_points[-1] - _x_points[0]
                    w_margin = 0
            else: # ?% 인접
                if _num_char == 1 and _widths[-1] > 50:
                    w = _widths[-1]
                    w_margin = -31
                else:
                    w = _x_points[-1] - _x_points[0] + _widths[-1] // 2
                    w_margin = 0
            
            _dst = crop_by_bounding_box(self.cell_image.copy(), x, y, w, h, y_offset=8, w_margin=w_margin)
            _dst = add_padding(_dst, [4, 20])
            
            return _dst
    
    def get_various_images(self, src):
        if len(self.choice_string_operator) == 0:
            _dst = cv.morphologyEx(src.copy(), op=cv.MORPH_DILATE, kernel=np.ones([2, 2]))
            _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
            return [{"image": _dst, "str": None}]
        _process_images = []
        _parameters = {"dilate3x3" : {"op": cv.MORPH_DILATE, "kernel": np.ones([2, 2])},
                       "dilate2x2" : {"op": cv.MORPH_ERODE, "kernel": np.ones([2, 2])},
                       "open2x2"   : {"op": cv.MORPH_OPEN, "kernel": np.ones([2, 2])},
                       "close3x3"  : {"op": cv.MORPH_CLOSE, "kernel": np.ones([3, 3])}}
        
        _src = src.copy()
        for idx, (_key, _param) in enumerate(_parameters.items()):
            if _key in self.choice_string_operator:
                _dst = cv.morphologyEx(_src, op=_param["op"], kernel=_param["kernel"])
                _dst = threshold(_dst, thresh_type=cv.THRESH_BINARY + cv.THRESH_OTSU, blur_size=[7,7])
                _process_images.append({"image": _dst, "str": None})

        return _process_images

    def get_char_bounding_box(self, src, min_width=None):
        _dst = threshold(src.copy(), thresh_type=cv.THRESH_BINARY_INV)

        _dst = cv.erode(_dst, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv.findContours(_dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]
        if min_width is not None:
            bounding_boxes = [box for box in bounding_boxes if box[2] > min_width]  # box[2]가 w (너비)를 나타냄

        return bounding_boxes

    def get_char_images(self, src):
        _bbox_list = self.get_char_bounding_box(src.copy(), 10)

        if self.num_char == 2 and len(_bbox_list) != 2:
            if len(_bbox_list) == 1:
                if _bbox_list[0][2] > 32:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+w//2, y, w//2, h], [x, y, w//2, h]]
                else:
                    logger.error(f"op1-1: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            elif len(_bbox_list) == 3:
                print(_bbox_list, self.num_char)
                plt.imshow(self.rm_percentile_image)
                plt.show()
            else:
                logger.error(f"op1 : _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                return []
        elif self.num_char == 3 and len(_bbox_list) != 3:
            if len(_bbox_list) == 1:
                if _bbox_list[0][2] > 64:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+2*(w//3), y, w//3, h], [x+w//3, y, w//3, h], [x, y, w//3, h]]
                else:
                    logger.error(f"op2 : _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            elif len(_bbox_list) == 2:
                if _bbox_list[0][2] > 32:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+w//2, y, w//2, h], [x, y, w//2, h], _bbox_list[1]]
                elif _bbox_list[1][2] > 32:
                    x, y, w, h = _bbox_list[1]
                    _bbox_list = [_bbox_list[0], [x+w//2, y, w//2, h], [x, y, w//2, h]]
                else:
                    logger.error(f"op3: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            else:
                logger.error(f"op4: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                return []

        _bbox_list.sort(key=lambda box: box[0], reverse=True)

        _temp_images = []
        for i, (_x, _y, _w, _h) in enumerate(_bbox_list, start=1):
            _dst = crop_by_bounding_box(src.copy(), _x, _y, _w, _h)
            _dst = add_padding(_dst, [10, 10])
            _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel=np.ones([2,2]))     
            _temp_images.append({"image": _dst})

        return _temp_images

    def display(self):
        if len(self.process_images) == 0:
            fig, axs = plt.subplots(1, 2, figsize=(6, 4))
            axs[0].imshow(self.cell_image, cmap="gray")
            axs[0].set_title(f"원본 이미지: {self.cell_image.shape}")
            axs[0].axis("on")
            axs[1].imshow(self.rm_percentile_image, cmap="gray")
            axs[1].set_title(f"% 제거 이미지: {self.cell_image.shape}")
            axs[1].axis("on")
        else:
            n_col = 2 + len(self.process_images) + len(self.char_images)
            fig, axs = plt.subplots(1, n_col, figsize=(2*(n_col), 4))
            axs[0].imshow(self.cell_image, cmap="gray")
            axs[0].set_title(f"원본 이미지: {self.cell_image.shape}")
            axs[0].axis("on")
            axs[1].imshow(self.rm_percentile_image, cmap="gray")
            axs[1].set_title(f"% 제거 이미지: {self.cell_image.shape}")
            axs[1].axis("on")
            for idx in range(0, len(self.process_images)):
                axs[idx+2].imshow(self.process_images[idx]["image"], cmap="gray")
                axs[idx+2].set_title(f"{self.process_images[idx]['str']}")
                axs[idx+2].axis("off")
    
            _images = self.char_images[::-1]
            for jdx in range(len(_images)):
                current_idx = len(self.process_images) + 2 + jdx
                if current_idx >= n_col:
                    break
                axs[current_idx].imshow(_images[jdx]["image"], cmap="gray")
                axs[current_idx].set_title(f"{_images[jdx]['str']}")
                axs[current_idx].axis("off")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        msg = "S.D.T 검증 알고리즘 - Percentile\n"
        if self.is_blank:
            msg += " - [이 미 지] 공백\n"
        else:
            msg += f" - [이 미 지] 입력된 문자열 예상 개수 (Pixel): {self.num_char} ({self.is_blank} / {self.string_width})\n"
        return msg

    def __call__(self):
        if self.is_blank:
            return ""
        _candidate_str_list = []

        tesseract_config = {"1-str": {"1th": '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789',},
                            "2-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"},
                            "3-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=0",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=0",
                                    "3th": "--psm 10 --oem 3 -c tessedit_char_whitelist=1"}}
        config = tesseract_config[f"{self.num_char}-str"]
        _str = ""
        _str_1th = ""
        if self.num_char == len(self.char_images):
            for idx, _temp in enumerate(self.char_images):
                _image = _temp["image"]
                _temp_str = tesseract_ocr(_image, config = config[f"{idx + 1}th"])
                self.char_images[idx]["str"] = _temp_str
                _str = _temp_str + _str 
            _str_1th = _str[-1] if len(_str) > 0 else ""
            _candidate_str_list.append(_str)

        for idx, _temp in enumerate(self.process_images):
            _image = _temp["image"]
            _sp_str = check_special_char(_image)
            _str = tesseract_ocr(_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            if len(_str) > 1 and _str[-1] != _str_1th:
                _str = _str[:-1] + _str_1th
            elif _sp_str is not None:
                _str = _sp_str
            self.process_images[idx]["str"] = _str
            _candidate_str_list.append(_str)

        return _candidate_str_list

class VerificationSDTStimulus:
    def __init__(self, cell_image:np.ndarray, choice_string_operator, off_char_checker, off_remove_symbol):
        self.cell_image = cell_image
        self.choice_string_operator = choice_string_operator
        self.off_char_checker = off_char_checker
        self.off_remove_symbol = off_remove_symbol

        self.rm_dB_image = self.remove_dB_str()
        self.is_blank = is_blank_image(self.rm_dB_image)
    
        self.process_images = []
        self.string_width = 0
        self.num_char = 0
        self.char_images = []

        if not self.is_blank:
            self.process_images = self.get_various_images(self.rm_dB_image.copy())
            self.string_width = get_string_width(self.rm_dB_image.copy())
            self.num_char = self.get_num_char()
            if not self.off_char_checker:
                self.char_images = self.get_char_images(src=self.rm_dB_image.copy())

    def get_num_char(self):
        if 0 < self.string_width <= 32:
            return 1
        elif 32 < self.string_width <= 62: # 두자리 숫자
            return 2
        elif 62 < self.string_width: # 세자리 숫자
            return 3

    def remove_dB_str(self):
        if self.off_remove_symbol:
            return self.cell_image.copy()
        else:
            _dst = self.cell_image.copy()
            _num_char, _widths, _x_points = get_characters_width(_dst)

            is_overlap = False
            if _num_char == 0:
                return np.zeros(_dst.shape)
            elif 0 < _num_char < 3: # ?dB # ?d B / d B
                if _widths[0] > 28:
                    is_overlap = True
                else: # d B 만 존재 -> 공백
                    return np.zeros(_dst.shape)
            else: # ? d B  / ? ?d B
                if _widths[-2] > 28: # ? d 안 겹침
                    is_overlap = True

            x = _x_points[0]
            y = 0
            h = _dst.shape[0]

            if is_overlap: # ?d 겹침
                if _num_char == 1: # ?dB
                    w = _widths[0] // 3 * 2
                    w_margin = 0
                elif _num_char == 2: # ?d B
                    w = _widths[0]
                    w_margin = -8
                elif _num_char == 3: # ? ?d B ? ??d B
                    if _widths[1] > 50:
                        w = _x_points[-1] - _x_points[0]
                        w_margin = -20
                    else:
                        w = _x_points[-2] - _x_points[0] + _widths[-2] // 2
                        w_margin = 5
                else: # ? ? ?d B
                    w = _x_points[-2] - _x_points[0] + _widths[-2]
                    w_margin = 0
            else:        
                w = _x_points[-2] - _x_points[0]
                w_margin = -8

            _dst = crop_by_bounding_box(_dst, x, y, w, h, w_margin=w_margin)
            _dst = add_padding(_dst, [20, 20])

            return _dst

    def get_various_images(self, src):
        if len(self.choice_string_operator) == 0:
            _dst = cv.morphologyEx(src.copy(), op=cv.MORPH_DILATE, kernel=np.ones([2, 2]))
            _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
            _dst = add_padding(_dst, [30, 0])
            return [{"image": _dst, "str": None}]
        _process_images = []
        _parameters = {"dilate3x3" : {"op": cv.MORPH_DILATE, "kernel": np.ones([2, 2])},
                       "dilate2x2" : {"op": cv.MORPH_ERODE, "kernel": np.ones([2, 2])},
                       "open2x2"   : {"op": cv.MORPH_OPEN, "kernel": np.ones([2, 2])},
                       "close3x3"  : {"op": cv.MORPH_CLOSE, "kernel": np.ones([3, 3])}}
        
        _src = cv.morphologyEx(src.copy(), op=cv.MORPH_DILATE, kernel=np.ones([2,2]))
        for _key, _param in _parameters.items():
            if _key in self.choice_string_operator:
                _dst = cv.morphologyEx(_src, op=_param["op"], kernel=_param["kernel"])
                _process_images.append({"image": _dst, "str": None})

        return _process_images

    def get_char_bounding_box(self, src, min_width=None):
        _dst = threshold(src.copy(), thresh_type=cv.THRESH_BINARY_INV)

        _dst = cv.erode(_dst, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv.findContours(_dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

        if min_width is not None:
            bounding_boxes = [box for box in bounding_boxes if box[2] > min_width]  # box[2]가 w (너비)를 나타냄

        return bounding_boxes

    def get_char_images(self, src):
        _bbox_list = self.get_char_bounding_box(src.copy(), 5)

        if self.num_char == 2 and len(_bbox_list) != 2:
            if len(_bbox_list) == 1:
                if _bbox_list[0][2] > 32:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+w//2, y, w//2, h], [x, y, w//2, h]]
                else:
                    logger.error(f"op1-1: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            else:
                logger.error(f"op1 : _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                return []
        elif self.num_char == 3 and len(_bbox_list) != 3:
            if len(_bbox_list) == 1:
                if _bbox_list[0][2] > 64:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+2*(w//3), y, w//3, h], [x+w//3, y, w//3, h], [x, y, w//3, h]]
                else:
                    logger.error(f"op2 : _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            elif len(_bbox_list) == 2:
                if _bbox_list[0][2] > 32:
                    x, y, w, h = _bbox_list[0]
                    _bbox_list = [[x+w//2, y, w//2, h], [x, y, w//2, h], _bbox_list[1]]
                elif _bbox_list[1][2] > 32:
                    x, y, w, h = _bbox_list[1]
                    _bbox_list = [_bbox_list[0], [x+w//2, y, w//2, h], [x, y, w//2, h]]
                else:
                    logger.error(f"op3: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                    return []
            else:
                logger.error(f"op4: _bbox >= {len(_bbox_list)} 와 num_char =={self.num_char} ")
                return []
            
        _bbox_list.sort(key=lambda box: box[0], reverse=True)

        _temp_images = []
        for i, (_x, _y, _w, _h) in enumerate(_bbox_list, start=1):
            _dst = crop_by_bounding_box(src, _x, _y, _w, _h)
            _dst = add_padding(_dst, [10, 10])
            _dst = threshold(_dst,  cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[5,5])
            _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel=np.ones([2,2]))
            _temp_images.append({"image": _dst})

        return _temp_images

    def display(self):
        if len(self.process_images) == 0:
            fig, axs = plt.subplots(1, 2, figsize=(4, 4))
            axs[0].imshow(self.cell_image, cmap="gray")
            axs[0].set_title(f"원본 이미지: {self.cell_image.shape}")
            axs[0].axis("on")
            axs[1].imshow(self.rm_dB_image, cmap="gray")
            axs[1].set_title(f"dB 제거 이미지: {self.cell_image.shape}")
            axs[1].axis("on")
        else:
            n_col = 2 + len(self.process_images) + len(self.char_images)
            fig, axs = plt.subplots(1, n_col, figsize=(2*(n_col), 4))
            axs[0].imshow(self.cell_image, cmap="gray")
            axs[0].set_title(f"원본 이미지: {self.cell_image.shape}")
            axs[0].axis("on")
            axs[1].imshow(self.rm_dB_image, cmap="gray")
            axs[1].set_title(f"dB 제거 이미지: {self.cell_image.shape}")
            axs[1].axis("on")
            for idx in range(0, len(self.process_images)):
                axs[idx+2].imshow(self.process_images[idx]["image"], cmap="gray")
                axs[idx+2].set_title(f"{self.process_images[idx]['str']}")
                axs[idx+2].axis("off")
    
            _images = self.char_images[::-1]
            for jdx in range(len(_images)):
                current_idx = len(self.process_images) + 2 + jdx
                if current_idx >= n_col:
                    break
                axs[current_idx].imshow(_images[jdx]["image"], cmap="gray")
                axs[current_idx].set_title(f"{_images[jdx]['str']}")
                axs[current_idx].axis("off")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        msg = "S.D.T 검증 알고리즘 - remove dB\n"
        if self.is_blank:
            msg += " - [이 미 지] 공백\n"
        else:
            msg += f" - [이 미 지] 입력된 문자열 예상 개수 (Pixel): {self.num_char} ({self.is_blank} / {self.string_width})\n"
        return msg

    def __call__(self):
        if self.is_blank:
            return ""
        _candidate_str_list = []

        tesseract_config = {"1-str": {"1th": '--psm 10 --oem 3 -c tessedit_char_whitelist=05',},
                            "2-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=05",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"},
                            "3-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=05",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789",
                                    "3th": "--psm 10 --oem 3 -c tessedit_char_whitelist=1-"}}
        config = tesseract_config[f"{self.num_char}-str"]
        _str = ""
        _str_1th = ""
        
        if self.num_char == len(self.char_images):
            for idx, _temp in enumerate(self.char_images):
                _image = _temp["image"]
                _temp_str = tesseract_ocr(_image, config = config[f"{idx + 1}th"])  # TODO: 여기 계속 에러남.... 
                if idx == 0:
                    _temp_str_1th = check_zero_five(_image)
                    if _temp_str != _temp_str_1th:
                        _temp_str = _temp_str_1th
                self.char_images[idx]["str"] = _temp_str
                _str = _temp_str + _str 
            _str_1th = _str[-1] if len(_str) > 0 else ""
            _candidate_str_list.append(_str)

        for idx, _temp in enumerate(self.process_images):
            _image = _temp["image"]
            _sp_str = check_special_char(_image)
            _str = tesseract_ocr(_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            if len(_str) > 1 and _str[-1] != _str_1th:
                _str = _str[:-1] + _str_1th
            elif len(_str) == 1 and self.num_char == 2:
                _str = _str + _str_1th
            elif _sp_str is not None:
                _str = _sp_str
            self.process_images[idx]["str"] = _str
            _candidate_str_list.append(_str)


        return _candidate_str_list