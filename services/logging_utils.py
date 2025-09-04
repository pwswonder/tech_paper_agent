# services/logging_utils.py
from __future__ import annotations
import logging, os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(
    name: Optional[str] = None, level: Optional[str] = None
) -> logging.Logger:
    """
    안전한 표준 로거 생성기.
    - 여러 번 호출해도 핸들러 중복 없음
    - LOG_LEVEL/LOG_FORMAT 환경변수 적용
    - uvicorn 하에서 중복 로그 방지(propagate=False)
    """
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = os.getenv("LOG_FORMAT", _DEFAULT_FORMAT)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    logger.propagate = False
    return logger
