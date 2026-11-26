import json
import logging

from src.domain.project import analyze


def calculate(rows, settings, rules=None):
    try:
        result = analyze(rows, settings, rules)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError as error:
        return None, str(error)
    except Exception:
        logging.getLogger(__name__).exception('计算运行失败')
        return None, '计算遇到内部错误，请检查台账后重试。'
    return result, None
