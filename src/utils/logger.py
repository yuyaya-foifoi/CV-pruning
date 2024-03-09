import logging
import os


def setup_logger(save_dir: str):
    log_filename = os.path.join(save_dir, "log.txt")

    # Loggerを設定
    logger = logging.getLogger()

    # ログレベルをDEBUGに設定（すべてのログを取得）
    logger.setLevel(logging.DEBUG)

    # フォーマッタを作成（ログメッセージの形式を設定）
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # ログをコンソールに出力するハンドラを作成
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # ログをファイルに出力するハンドラを作成
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # ハンドラをloggerに追加
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
