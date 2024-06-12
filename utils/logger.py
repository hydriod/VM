from __future__ import annotations
from pathlib import Path
import logging


def setup_logger(
    filepath: Path | str = "./log/root.log",
    logger_name: str | None = None,
    level=logging.INFO
):
    '''
    set up logger

    Arg
    ---
    filepath
        保存するログのパス
    '''
    # ディレクトリ作成
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath = str(filepath)

    # フォーマット指定
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    # ファイルに出力
    fh = logging.FileHandler(filepath)
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    # stderrに出力
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
