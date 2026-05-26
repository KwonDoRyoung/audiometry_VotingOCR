# -*-coding: utf-8
from typing import List
import logging
import cv2 as cv
import numpy as np
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .vision import *
from .utils import calculate_string_similarity
from .utils import crop_and_trim, find_table_grid_points

from .utils.utils import tesseract_ocr
from .utils.string import is_blank_image

logger = logging.getLogger(__name__)

class PTA:
    def __init__(self, image: np.ndarray, is_verification:bool, y1:int = 550, y2:int = 690, x1:int = 0, x2:int = 550, padding:int=-5, scale_factor:float=4):
        self.is_verification = is_verification
        self.region_coordinate = [y1, y2, x1, x2]
        self.region_image, self.cell_bounding_box_list = self.crop_region(image, y1, y2, x1, x2, padding, scale_factor)
        self.side_images = []
        self.conduction_images = []
        self.table_images = []
        self.base_column_list = ["Side", 
                            '청력(0.125kHz)-air', '청력(0.125kHz)-bone', 
                            '청력(0.25kHz)-air', '청력(0.25kHz)-bone',
                            '청력(0.5kHz)-air', '청력(0.5kHz)-bone',
                            '청력(0.75kHz)-air', '청력(0.75kHz)-bone',
                            '청력(1.0kHz)-air', '청력(1.0kHz)-bone',
                            '청력(1.5kHz)-air', '청력(1.5kHz)-bone',
                            '청력(2.0kHz)-air','청력(2.0kHz)-bone',
                            '청력(3.0kHz)-air', '청력(3.0kHz)-bone', 
                            '청력(4.0kHz)-air', '청력(4.0kHz)-bone', 
                            '청력(6.0kHz)-air', '청력(6.0kHz)-bone', 
                            '청력(8.0kHz)-air', '청력(8.0kHz)-bone',
                            '청력(12.0kHz)-air', '청력(12.0kHz)-bone',] 
        if self.is_verification:
            self.additional_column_list = ['voting_청력(0.125kHz)-air', 'voting_청력(0.125kHz)-bone', 
                                      'voting_청력(0.25kHz)-air', 'voting_청력(0.25kHz)-bone',
                                      'voting_청력(0.5kHz)-air', 'voting_청력(0.5kHz)-bone',
                                      'voting_청력(0.75kHz)-air', 'voting_청력(0.75kHz)-bone',
                                      'voting_청력(1.0kHz)-air', 'voting_청력(1.0kHz)-bone',
                                      'voting_청력(1.5kHz)-air', 'voting_청력(1.5kHz)-bone',
                                      'voting_청력(2.0kHz)-air','voting_청력(2.0kHz)-bone',
                                      'voting_청력(3.0kHz)-air', 'voting_청력(3.0kHz)-bone', 
                                      'voting_청력(4.0kHz)-air', 'voting_청력(4.0kHz)-bone', 
                                      'voting_청력(6.0kHz)-air', 'voting_청력(6.0kHz)-bone', 
                                      'voting_청력(8.0kHz)-air', 'voting_청력(8.0kHz)-bone',
                                      'voting_청력(12.0kHz)-air', 'voting_청력(12.0kHz)-bone']
        else:
            self.additional_column_list = []
        
    def crop_region(self, image, y1, y2, x1, x2, padding, scale_factor):
        _temp = crop_by_coordinate(image, y1, y2, x1, x2)
        _temp = crop_and_trim(_temp, padding=padding)

        _temp = resize_image(_temp, scale_factor_x=scale_factor, scale_factor_y=scale_factor)
        _temp = threshold(_temp, thresh_type=cv.THRESH_BINARY + cv.THRESH_OTSU)
        # 원본: threshold_w=15000, threshold_h = 75000
        # 4배율: threshold_w=15000*4, threshold_h = 75000*4
        _cell_bounding_box_list = find_table_grid_points(_temp, 
                                                         n_peak_x=13*2, 
                                                         n_peak_y=5*2,
                                                         threshold_w=15000 * scale_factor, 
                                                         threshold_h = 75000 * scale_factor)
        return _temp, _cell_bounding_box_list
    
    def ocr_conduction(self, x, y_points, w, h):
        _str_list = []
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image, x, y, w, h, x_offset=30)
                _dst = cv.morphologyEx(_dst, op=cv.MORPH_OPEN, kernel=np.ones([2, 2]))
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
                self.conduction_images.append({"image": _dst})

                _str = tesseract_ocr(_dst, config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABC")
                if _str not in ["AC", "BC"]:
                    _str = ""

                _str_list.append(_str)
            except Exception as e:
                logger.error(f"Error in processing bounding box ({y}, {w}, {h}): {e}", exc_info=True)
                raise

        if "BC" not in set(_str_list):
            _str_list = ["AC" if _str == "AC" else "BC" for _str in _str_list]
        elif "AC" not in set(_str_list):
            _str_list = ["BC" if _str == "BC" else "AC" for _str in _str_list]

        if any(_str not in ["AC", "BC"] for _str in _str_list):
            logger.error(f"Unexpected characters in conduction: {_str_list}")
            raise ValueError(f"Invalid characters in conduction: {_str_list}")
        
        for idx, _str in enumerate(_str_list):
            self.conduction_images[idx]["str"] = _str

        return _str_list

    def ocr_side(self, x,  y_points, w, h):        
        _str_list = []
        for y in y_points:
            try:
                _dst = crop_by_bounding_box(self.region_image, x, y, w, h, w_margin=-10)
                _dst = cv.morphologyEx(_dst, op=cv.MORPH_OPEN, kernel=np.ones([5, 5]))
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
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
    
    def ocr_table(self):
        _str_list = []

        for row_bounding_box_list in self.cell_bounding_box_list:
            _row_str_list = []
            _temp_row_images = []
            for x, y, w, h in row_bounding_box_list:
                try:
                    _dst = crop_by_bounding_box(self.region_image, x, y, w, h, 
                                                x_offset=3, y_offset = 3, 
                                                w_margin = -8, h_margin = -2)
                    _dst = cv.morphologyEx(_dst, op=cv.MORPH_OPEN, kernel=np.ones([2, 2]))
                    _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
                    _dst = add_padding(_dst, [20, 0])
                        
                    if is_blank_image(_dst): # 공백
                        _str = ""
                    else:
                        _str = tesseract_ocr(_dst, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789NR-')
                    _temp_row_images.append({"image": _dst, "str": _str})
                    _row_str_list.append(_str)
                    
                except Exception as e:
                    logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                    raise

            _str_list.append(_row_str_list)
            self.table_images.append(_temp_row_images)
        
        return _str_list

    def ocr_table_verification(self, choice_string_operator, off_char_checker):
        _str_list = []
        _percent_list = []
        for row_bounding_box_list in self.cell_bounding_box_list:
            _row_str_list = []
            _row_percent_list = []
            _temp_row_images = []
            for x, y, w, h in row_bounding_box_list:
                try:
                    _dst = crop_by_bounding_box(self.region_image, x, y, w, h, 
                                                x_offset=3, y_offset = 3, 
                                                w_margin = -8, h_margin = -2)
                    _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])

                    verification = VerificationPTA(cell_image=_dst, choice_string_operator=choice_string_operator, off_char_checker=off_char_checker)
                    # print(verification)
                    _candidate_str_list = verification()
                    # verification.display()
                    if _candidate_str_list == "": # 공백
                        _main_str = ""
                        _main_voting_percent = 100
                    else:
                        _main_str, _main_voting_percent = calculate_string_similarity(_candidate_str_list)
                    _temp_row_images.append({"image": _dst, "str": _main_str, "str_candidate": _candidate_str_list})
                    _row_str_list.append(_main_str)
                    _row_percent_list.append(_main_voting_percent)
                except Exception as e:
                    logger.error(f"Error in processing bounding box ({x}, {y}, {w}, {h}): {e}", exc_info=True)
                    raise
            _str_list.append(_row_str_list)
            _percent_list.append(_row_percent_list)
            self.table_images.append(_temp_row_images)
        
        return _str_list, _percent_list


    def __str__(self):
        mode = "검증 알고리즘 모드" if self.is_verification else "일반 모드"
        msg = f"P.T.A. OCR - {mode}\n"
        msg += f" - Table 좌표: {self.region_coordinate}\n"
        return msg

    def display(self):
        # 전체 Figure 생성
        fig = plt.figure(figsize=(18, 8))
        gs = gridspec.GridSpec(5, 12, figure=fig, height_ratios=[1, 0.7, 0.7, 0.7, 0.7], width_ratios=[0.7, 0.7] + [1.3]*10)

        # 첫 번째 행 전체를 하나의 이미지로 사용 (가로로 긴 이미지)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(self.region_image, cmap="gray")
        ax1.axis("off")

        # 첫 번째 열의 두 번째와 세 번째 행을 합쳐서 하나의 이미지
        ax2 = fig.add_subplot(gs[1:3, 0])
        ax2.imshow(self.conduction_images[0]["image"], cmap="gray")
        ax2.set_title(self.conduction_images[0]["str"])
        ax2.axis("off")

        # 첫 번째 열의 네 번째와 다섯 번째 행을 합쳐서 하나의 이미지
        ax3 = fig.add_subplot(gs[3:5, 0])
        ax3.imshow(self.conduction_images[1]["image"], cmap="gray")
        ax3.set_title(self.conduction_images[1]["str"])
        ax3.axis("off")

        # 두 번째 열의 두 번째, 세 번째, 네 번째, 다섯 번째 행에 각각 하나의 이미지
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(self.side_images[0]["image"], cmap="gray")
        ax4.set_title(self.side_images[0]["str"])
        ax4.axis("off")

        ax5 = fig.add_subplot(gs[2, 1])
        ax5.imshow(self.side_images[1]["image"], cmap="gray")
        ax5.set_title(self.side_images[1]["str"])
        ax5.axis("off")

        ax6 = fig.add_subplot(gs[3, 1])
        ax6.imshow(self.side_images[2]["image"], cmap="gray")
        ax6.set_title(self.side_images[2]["str"])
        ax6.axis("off")

        ax7 = fig.add_subplot(gs[4, 1])
        ax7.imshow(self.side_images[3]["image"], cmap="gray")
        ax7.set_title(self.side_images[3]["str"])
        ax7.axis("off")

        # 세 번째 열부터는 4x12 Grid 이미지로 구성
        for idx, i in enumerate(range(2, 12)):  # 세 번째 열부터 마지막 열까지 반복
            for jdx, j in enumerate(range(1, 5)):  # 두 번째 행부터 다섯 번째 행까지 반복
                ax = fig.add_subplot(gs[j, i])
                img_str = self.table_images[jdx][idx]
                ax.imshow(img_str["image"], cmap="gray")
                if self.is_verification:
                    _candidate = ', '.join(img_str["str_candidate"])
                    ax.set_title(f'{img_str["str"]}\n{_candidate}')
                else:
                    ax.set_title(f'{img_str["str"]}')
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def __call__(self, choice_string_operator, off_char_checker):
        t_x, y1, w, h = self.cell_bounding_box_list[0][0]
        _, y2, _, _ = self.cell_bounding_box_list[1][0]
        _, y3, _, _ = self.cell_bounding_box_list[2][0]
        _, y4, _, _ = self.cell_bounding_box_list[3][0]
        conducation_str = self.ocr_conduction(0, [y1, y3], w, 2*h)
        side_str = self.ocr_side(t_x - w, [y1, y2, y3, y4], w, h)

        if not self.is_verification:
            table_str = self.ocr_table()
            table_str_voting_percent = None
        else:
            table_str, table_str_voting_percent = self.ocr_table_verification(choice_string_operator=choice_string_operator, off_char_checker=off_char_checker)

        column_list = ["125", "250", "500", "750", "1000", "1500", "2000", "3000", "4000", "6000", "8000", "12000"]
        pta_data = pd.DataFrame(table_str, columns=column_list)
        if self.is_verification is not None:
            pta_voting_data = pd.DataFrame(table_str_voting_percent, columns=[f"voting_{col}" for col in column_list])
            pta_data = pd.concat([pta_data, pta_voting_data], axis=1)

        pta_data["Side"] = side_str    
        pta_data["Cond"] = [conducation_str[0], conducation_str[0], conducation_str[1], conducation_str[1]]    
        pta_data = pta_data[["Cond"]+["Side"] + column_list + [f"voting_{col}" for col in column_list]]

        format_info = []
        right_side = {"Side": "R"}
        left_side = {"Side": "L"}
        for idx, row in pta_data.iterrows():
            cond = "air" if row["Cond"] == "AC" else "bone"
            side = row["Side"]
            hz_value = {}
            for hz in column_list:
                if hz == "125":
                    rename_hz = "청력(0.125kHz)"
                elif hz == "250":
                    rename_hz = "청력(0.25kHz)"
                elif hz == "500":
                    rename_hz = "청력(0.5kHz)"
                elif hz == "750":
                    rename_hz = "청력(0.75kHz)"
                elif hz in ["1000", "1500", "2000", "3000", "4000", "6000", "8000", "12000"]:
                    rename_hz = f"청력({int(hz)/1000}kHz)"
                hz_value[f"{rename_hz}-{cond}"] = row[hz]
                if table_str_voting_percent is not None:
                    hz_value[f"voting_{rename_hz}-{cond}"] = row[f"voting_{hz}"]
            if side == "R":
                right_side.update(hz_value)
            elif side == "L":
                left_side.update(hz_value)
        format_info.append(right_side)
        format_info.append(left_side)

        new_pta_df = pd.DataFrame.from_dict(format_info)

        new_pta_df = new_pta_df[self.base_column_list + self.additional_column_list]

        return new_pta_df

class VerificationPTA:
    def __init__(self, cell_image: np.ndarray, choice_string_operator: list[str], off_char_checker: bool):
        self.cell_image = cell_image
        self.is_blank = is_blank_image(cv.bitwise_not(cell_image))
        self.process_images = []
        self.string_width = 0
        self.num_char = 0
        self.char_images = []

        self.choice_string_operator = choice_string_operator # ["open2x2", "dilate3x3", "dilate2x2", "close3x3"]
        self.off_char_checker = off_char_checker

        if not self.is_blank:
            self.process_images = self.get_various_images()
            self.string_width = self.get_string_width()
            self.num_char = self.get_num_char()
            if not self.off_char_checker:
                self.char_images = self.get_char_images()

    def get_various_images(self):
        if len(self.choice_string_operator) == 0:
            _dst = cv.morphologyEx(self.cell_image.copy(), op=cv.MORPH_OPEN, kernel=np.ones([2, 2]))
            _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU, blur_size=[7, 7])
            _dst = add_padding(_dst, [20, 0])
            return [{"image": _dst, "str": None}]
        _process_images = []
        _parameters = {"dilate3x3" : {"op": cv.MORPH_DILATE, "kernel": np.ones([3, 3]), "padding":[20, 0]},
                       "dilate2x2" : {"op": cv.MORPH_DILATE, "kernel": np.ones([2, 2]), "padding":[20, 0]},
                       "open2x2"   : {"op": cv.MORPH_OPEN, "kernel": np.ones([2, 2]), "padding":[20, 0]},
                       "close3x3"  : {"op": cv.MORPH_CLOSE, "kernel": np.ones([3, 3]), "padding":[20, 0]}}
        
        for idx, (_key, _param) in enumerate(_parameters.items()):
            if _key in self.choice_string_operator:
                _dst = cv.morphologyEx(self.cell_image, op=_param["op"], kernel=_param["kernel"])
                _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)
                _dst = add_padding(_dst, padding=_param["padding"])
                _process_images.append({"image": _dst, "str": None})

        return _process_images
    
    def get_char_bounding_box(self):
        _, _dst = cv.threshold(self.cell_image, 150, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        _dst = cv.erode(_dst, kernel, iterations=1)
        _dst = cv.dilate(_dst, kernel, iterations=1)

        contours, _ = cv.findContours(_dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_characters = len(contours)

        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

        return num_characters, bounding_boxes

    def get_char_images(self):
        _num_char, _bbox_list = self.get_char_bounding_box()
        if self.num_char == 2:
            if _num_char == 1 and _bbox_list[0][2] > 32:
                x, y, w, h = _bbox_list[0]
                _bbox_list = [[x+w//2, y, w//2, h], [x, y, w//2, h]]
        _bbox_list.sort(key=lambda box: box[0], reverse=True)
        _temp_images = []
        for i, (_x, _y, _w, _h) in enumerate(_bbox_list, start=1):
            _dst = crop_by_bounding_box(self.cell_image, _x, _y, _w, _h)
            _dst = add_padding(_dst, [20, 20])
            _dst = cv.morphologyEx(_dst, op=cv.MORPH_DILATE, kernel= np.ones([3, 3]))
            _dst = threshold(_dst, cv.THRESH_BINARY+cv.THRESH_OTSU)
            _temp_images.append({"image": _dst})

        return _temp_images
    
    def __str__(self):
        msg = "P.T.A. 검증 알고리즘\n"
        if self.is_blank:
            msg += " - [이 미 지] 공백\n"
        else:
            msg += f" - [이 미 지] 입력된 문자열 예상 개수 (Pixel): {self.num_char} ({self.is_blank} / {self.string_width})\n"
        return msg

    def get_string_width(self):
        width_dist = np.sum(cv.bitwise_not(self.cell_image), axis=0)

        non_zero_indices = np.nonzero(width_dist)[0]

        if non_zero_indices.size == 0:
            # 양수 값이 없는 경우 (즉, 이미지가 비어 있는 경우)
            return 0

        # l_idx는 첫 번째 양수 인덱스, r_idx는 마지막 양수 인덱스입니다.
        l_idx = non_zero_indices[0]
        r_idx = non_zero_indices[-1]

        return r_idx - l_idx
    
    def get_num_char(self):
        if 0 < self.string_width <= 32:
            return 1
        elif 32 < self.string_width <= 62: # 두자리 숫자
            return 2
        elif 62 < self.string_width: # 세자리 숫자
            return 3
        
    def display(self):
        if len(self.process_images) == 0:
            fig, axs = plt.subplots(1, 1, figsize=(4, 4))
            axs.imshow(self.cell_image, cmap="gray")
            axs.set_title(f"원본 이미지: {self.cell_image.shape}")
        else:
            n_col = len(self.process_images) + 1 + self.num_char
            fig, axs = plt.subplots(1, n_col, figsize=(2*(n_col), 4))
            axs[0].imshow(self.cell_image, cmap="gray")
            axs[0].set_title(f"원본 이미지: {self.cell_image.shape}")
            for idx in range(0, len(self.process_images)):
                axs[idx+1].imshow(self.process_images[idx]["image"], cmap="gray")
                axs[idx+1].set_title(f"{self.process_images[idx]['str']}")
                axs[idx+1].axis("off")
            _images = self.char_images[::-1]
            for jdx, idx in enumerate(range(len(self.process_images), n_col - 1)):
                axs[idx+1].imshow(_images[jdx]["image"], cmap="gray")
                axs[idx+1].set_title(f"{_images[jdx]['str']}")
                axs[idx+1].axis("off")

        plt.tight_layout()    
        plt.show()

    def __call__(self):
        if self.is_blank:
            return ""
        
        _candidate_str_list = []
        for idx, _temp in enumerate(self.process_images):
            _image = _temp["image"]
            _str = tesseract_ocr(_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789NR-')
            if self.num_char == 1:
                if _str == "" and 0 < self.string_width < 30:
                    _str = "-"
            elif self.num_char == 2:
                if _str == "5":
                    _str = "-5" if self.string_width > 30 else "5"
                elif _str == "0":
                    _str = "30"
                elif _str == "7":
                    _str = "75"
                elif _str == "2":
                    _str = "25"

            self.process_images[idx]["str"] = _str
            _candidate_str_list.append(_str)

        tesseract_config = {"1-str": {"1th": '--psm 10 --oem 3 -c tessedit_char_whitelist=05-',},
                            "2-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=05R",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789N-"},
                            "3-str": {"1th": "--psm 10 --oem 3 -c tessedit_char_whitelist=05",
                                    "2th": "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789",
                                    "3th": "--psm 10 --oem 3 -c tessedit_char_whitelist=1-"}}
        config = tesseract_config[f"{self.num_char}-str"]
        _str = ""
        for idx, _temp in enumerate(self.char_images):
            _image = _temp["image"]
            _temp_str = tesseract_ocr(_image, config = config[f"{idx + 1}th"])
            self.char_images[idx]["str"] = _temp_str
            _str = _temp_str + _str 
        _candidate_str_list.append(_str)

        return _candidate_str_list
    