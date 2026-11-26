"""Domain entry point for the lacquer drying workbench."""

from math import isfinite

from .fixtures import demo_rows as demo_rows
from .model import chronological_split, fit_gamma
from .records import FIELDS, normalize_rows as normalize_rows
from .reporting import build_report


SPEC = {
    "columns": list(FIELDS),
    "fields": FIELDS,
    "settings": [
        {"key": "holdout_percent", "label": "末期作业日留出比例（%）", "default": 25, "min": 10, "max": 40, "step": 5},
        {"key": "alpha", "label": "Gamma 正则系数", "default": 0.01, "min": 0.001, "max": 1, "step": 0.001},
    ],
    "modules": {"datasets": "补漆来源台账", "runs": "拟合与时序留出", "results": "干燥估时与排班", "reviews": "排班复核与导出"},
    "limitation": "估时适用于同类漆料与干燥判定条件。超训练范围的预测须现场复核，完工仍以实测确认。",
    "dataset_name": "雕版补漆工单_202611.csv",
    "recommendation": "优先核对估时超过预留的工单，按配漆批次发放等候与排班清单。",
}


def _settings(settings: dict) -> dict:
    values = {}
    for control in SPEC["settings"]:
        key = control["key"]
        raw = settings.get(key, control["default"])
        try:
            valid_type = isinstance(raw, (int, float, str)) and not isinstance(raw, bool)
            value = float(raw) if valid_type else float("nan")
        except (ValueError, OverflowError):
            value = float("nan")
        if not isfinite(value) or not control["min"] <= value <= control["max"]:
            raise ValueError(f"{control['label']}须为 {control['min']:g} 至 {control['max']:g} 之间的有限数字。")
        values[key] = value
    return values


def analyze(rows: list[dict], settings: dict) -> dict:
    options = _settings(settings)
    split = chronological_split(normalize_rows(rows), options["holdout_percent"])
    return build_report(split, fit_gamma(split.train, options["alpha"]))
