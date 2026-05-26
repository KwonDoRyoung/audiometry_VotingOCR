# -*-coding: utf-8
import logging
import cv2 as cv
import numpy as np

from ..vision import *

logger = logging.getLogger(__name__)

def calculate_string_similarity(strings, num_char=None):
    if "무" in strings:
        return "무", 50
    if num_char is not None and num_char > 0:
        _strings = [item for item in strings if item != "" and len(item) == num_char ]
    else:
        _strings = strings

    if len(_strings) == 0:
        if "9" in strings and num_char == 2:
            return "96", 1
        elif "15" in strings and num_char == 3:
            return "100", 1
        else:
            frequency = {s: strings.count(s) for s in set(strings)}
            # 빈도가 가장 높은 문자열 찾기
            max_count = max(frequency.values())
            most_similar_strings = [s for s, count in frequency.items() if count == max_count][0]
            return most_similar_strings, 0

    # 문자열의 빈도 계산
    frequency = {s: _strings.count(s) for s in set(_strings)}

    # 빈도가 가장 높은 문자열 찾기
    max_count = max(frequency.values())
    most_similar_strings = [s for s, count in frequency.items() if count == max_count][0]
    similarity_percentage = (max_count / len(strings)) * 100
    # 모두 다르면 0% 반환
    if max_count == 1:
        return most_similar_strings, similarity_percentage  # 유사한 문자열이 없으면 None과 0% 반환

    # 유사도가 가장 높은 문자열 찾기
    return most_similar_strings, similarity_percentage


def is_blank_image(src):
    return True if np.sum(src) == 0 else False


def get_characters_width(src):
    src = threshold(src, thresh_type=cv.THRESH_BINARY_INV)
    width_dist = np.sum(src, axis=0)
    in_text = False
    x_points = []
    widths = []

    for idx in range(len(width_dist)):
        if not in_text and width_dist[idx] > 0:
            # text 진입 시점
            in_text = True
            x_points.append(idx)
        elif in_text and width_dist[idx] == 0:
            in_text = False
            widths.append(idx-x_points[-1])

    if in_text:
        widths.append(len(width_dist) - x_points[-1])

    num_char = len(widths)

    return num_char, widths, x_points


def calculate_string_width(src, is_inv=False):
    if not isinstance(src, np.ndarray):
        logging.error(f"Expected src to be np.ndarray, but got {type(src)} instead.")
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)} instead.")
    
    if src is None or src.size == 0:
        logging.error("Empty or invalid image input.")
        raise ValueError("Empty or invalid image input.")
    
    if len(src.shape) == 3:  # 컬러 이미지일 경우만 그레이스케일로 변환
        src = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

    if is_inv:
        src = cv.bitwise_not(src)
    
    _w = src.shape[1]
    _w_dist = np.sum(src, axis=0)

    l_idx = None
    for i in range(_w):
        if _w_dist[i] > 0:
            l_idx = i
            break
    
    r_idx = None
    for i in range(_w - 1, -1, -1):
        if _w_dist[i] > 0:
            r_idx = i
            break
    
    if l_idx is None or r_idx is None:
        logging.debug("Could not find valid boundaries for the string.")
        return 0

    width = 0 if r_idx - l_idx < 0 else r_idx - l_idx
    logging.debug(f"String width calculated: {width}")
    
    return width


def count_characters_in_image(src, reverse = True):
    if not isinstance(src, np.ndarray):
        logging.error(f"Expected src to be np.ndarray, but got {type(src)} instead.")
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)} instead.")
    
    # 이미지가 비어있는지 확인
    if src is None or src.size == 0:
        logging.error("Empty or invalid image input.")
        raise ValueError("Empty or invalid image input.")
    try:
        if len(src.shape) == 3:
            gray_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        elif len(src.shape) == 2:
            gray_image = src

        _, binary_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)  # 3x3 커널 생성
        eroded_image = cv.erode(binary_image, kernel, iterations=1)  # 침식
        dilated_image = cv.dilate(eroded_image, kernel, iterations=1)  # 팽창

        contours, _ = cv.findContours(dilated_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_characters = len(contours)

        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

        bounding_boxes.sort(key=lambda box: box[0], reverse=reverse)

        return num_characters, bounding_boxes
    except cv.error as e:
        logging.error(f"OpenCV error: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise

def count_characters_and_widths(src, reverse = True):
    if not isinstance(src, np.ndarray):
        logging.error(f"Expected src to be np.ndarray, but got {type(src)} instead.")
        raise TypeError(f"Expected src to be np.ndarray, but got {type(src)} instead.")
    
    # Check for empty or invalid image input
    if src is None or src.size == 0:
        logging.error("Empty or invalid image input.")
        raise ValueError("Empty or invalid image input.")

    try:
        gray_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        _, binary_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        eroded_image = cv.erode(binary_image, kernel, iterations=1)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=1)

        contours, _ = cv.findContours(dilated_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        num_characters = len(contours)

        bounding_boxes = [cv.boundingRect(cnt) for cnt in contours]

        bounding_boxes.sort(key=lambda box: box[0], reverse=reverse)

        character_widths = [box[2] for box in bounding_boxes]

        return num_characters, character_widths, bounding_boxes
    except cv.error as e:
        logging.error(f"OpenCV error: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise
