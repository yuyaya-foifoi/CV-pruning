from datetime import datetime


def get_current_datetime_for_path():
    now = datetime.now()

    # 日時を文字列に変換 ('YYYY_MM_DD_HH_MM_SS' format)
    return now.strftime("%Y_%m_%d_%H_%M_%S")
