# -*-coding: utf-8
import sys

sys.path.append("../../ocr_audiometry/")
from ocr.utils import load_workspace_path

workspace_path = load_workspace_path()

import logging
import os
import argparse
import cv2 as cv
import pandas as pd
import concurrent.futures
from tqdm import tqdm

from ocr import run_ocr_PInfo, PTA, SRT, SDT


def ocr(**kwargs):
    print(kwargs)
    is_verification = kwargs.get("is_verification")
    display = kwargs.pop("display")
    data_root_path = kwargs.pop("data_root_path")
    file_name = kwargs.get("file_name")
    file_path = os.path.join(data_root_path, f"{file_name}.jpg")

    choice_string_operator = kwargs.pop("choice_string_operator")
    off_char_checker = kwargs.pop("off_char_checker")
    off_remove_symbol = kwargs.pop("off_remove_symbol")

    data_dict = kwargs
    data_dict["ref"] = ""

    if not os.path.exists(file_path):
        logging.error(f"파일이 존재하지 않습니다. {file_name}")
        data_dict["ref"] = "파일이 존재하지 않음"
        return pd.DataFrame([data_dict])
    
    try:
        img = cv.imread(file_path)
    except:
        logging.error(f"Failed to load image: {file_path}")
        data_dict["ref"] = "파일이 열리지 않음"
        return pd.DataFrame([data_dict])
    
    if img is None:
        logging.error(f"Failed to load image: {file_path}")
        data_dict["ref"] = "파일이 열리지 않음"
        return pd.DataFrame([data_dict])

    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    p_info_dict = run_ocr_PInfo(img, display=display)
    if p_info_dict is None:
        data_dict["ref"] = "환자 나이 & 날짜 에러"
        return pd.DataFrame([data_dict])
    else:
        data_dict.update(p_info_dict)

    new_data_dict = [data_dict.copy(), data_dict.copy()]
    new_data_dict[0]["Side"] = "R"
    new_data_dict[1]["Side"] = "L"
    data_df = pd.DataFrame(new_data_dict)

    ocr_pta = PTA(img, is_verification=is_verification)
    pta_df = ocr_pta(choice_string_operator=choice_string_operator, off_char_checker=off_char_checker)

    ocr_srt = SRT(img, is_verification=is_verification)
    srt_df = ocr_srt(choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)

    ocr_sdt = SDT(img, is_verification=is_verification)
    sdt_df = ocr_sdt(choice_string_operator=choice_string_operator, off_char_checker=off_char_checker, off_remove_symbol=off_remove_symbol)

    df_merged = data_df.merge(pta_df, on="Side").merge(srt_df, on='Side').merge(sdt_df, on='Side')

    return df_merged



def main(args):
    info_path = args.info_path
    if "~/" in info_path:
        info_path = info_path.replace("~/", workspace_path)

    data_root_path = args.data_root_path
    if "~/" in data_root_path:
        data_root_path = data_root_path.replace("~/", workspace_path)

    if args.sheet_name is not None:
        info = pd.read_excel(info_path, sheet_name=args.sheet_name)
    else:
        info = pd.read_excel(info_path)
    print(len(info))
    parameters = []
    for _, rows in info.iterrows():
        # hid = rows['등록번호'] # hid
        hid = rows['ID'] # hid
        # name = rows['환자명'] # name
        # name = rows["이름"]
        name = rows["Name"]
        # prescription_date = rows['처방일'] # p_date
        prescription_date = rows['p_date'] # p_date
        # image_file_name = rows['이미지번호'] # file_name
        # image_file_name = rows['이미지파일명'] # file_name
        image_file_name = rows['file_name'] # file_name
        checked = rows['Checked']
        if args.eval and checked != 1:
            continue
        parameters.append({"ID": hid,
                            "Name": name,
                            "p_date": prescription_date,
                            "file_name": image_file_name,
                            "data_root_path": data_root_path,
                            "is_verification": args.is_verification,
                            "choice_string_operator": [] if args.choice_string_operator is None else args.choice_string_operator,
                            "off_char_checker": args.off_char_checker,
                            "off_remove_symbol": args.off_remove_symbol,
                            "display": False})
    
    if args.debug:
        parameters = parameters[:int(len(parameters)*0.001)]
        print(f"Total number of images: {int(len(parameters)*0.1)}")
    else:
        print(f"Total number of images: {len(parameters)}")
    total_data = []  # 전체 데이터를 저장할 리스트
    max_workers = min(32, os.cpu_count() - 4) 
    try:
        # ProcessPoolExecutor 사용
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ocr, **kwargs) for kwargs in parameters]

            # 결과 처리
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    total_data.append(result)
                except Exception as e:
                    if args.debug:
                        print(f"Error occurred: {e}")
                    else:
                        logging.error(f"Error in OCR processing: {e}")

    except KeyboardInterrupt:
        print("Keyboard interrupt detected! Saving progress...")
    finally:
        # DataFrame 생성
        df = pd.concat(total_data, ignore_index=True)

        # 데이터 저장 경로 설정
        save_path = args.save_path.replace("~/", workspace_path) if "~/" in args.save_path else args.save_path
        df.to_excel(save_path, index=False)
        
        print(f"Data saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-path", type=str, required=True)
    parser.add_argument("--data-root-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--is-verification", action="store_true")
    parser.add_argument("--choice-string-operator", type=str, nargs="+")
    parser.add_argument("--off-char-checker", action="store_true")
    parser.add_argument("--off-remove-symbol", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    main(args)