"""干燥柜仪表校准规则：换表后按仪表与测读时段更正湿度读数。"""

from datetime import date
from math import isfinite
import re
from typing import Literal, TypedDict

from .records import BOUNDS, Row


class CalibrationRule(TypedDict):
    meter: str
    starts: str
    ends: str
    lower: float
    upper: float


RULE_FIELDS = {"meter", "starts", "ends", "lower", "upper"}
RULE_LABELS = {
    "meter": "干燥柜仪表编号",
    "starts": "测读时段起",
    "ends": "测读时段止",
    "lower": "湿度校正下限（百分点）",
    "upper": "湿度校正上限（百分点）",
}
MAX_RULES = 200
MAX_SHIFT = 30.0
BOUNDS_ORDER: tuple[Literal["lower"], Literal["upper"]] = ("lower", "upper")
_METER = re.compile(r"[^\W_][\w.-]{0,63}")


def _meter(value: object) -> str:
    if isinstance(value, str) and _METER.fullmatch(value.strip()):
        return value.strip()
    raise ValueError("干燥柜仪表编号须以字母或数字开头，且只可使用字母、数字、下划线、点或连字符，最多 64 字符。")


def _day(value: object, label: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        try:
            if date.fromisoformat(text).isoformat() == text:
                return text
        except ValueError:
            pass
    raise ValueError(f"{label}须为有效的 YYYY-MM-DD 日期。")


def _shift(value: object, label: str) -> float:
    try:
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        number = float(value) if valid_type else float("nan")
    except (ValueError, OverflowError):
        number = float("nan")
    if not isfinite(number) or not -MAX_SHIFT <= number <= MAX_SHIFT:
        raise ValueError(f"{label}须为 {-MAX_SHIFT:g} 至 {MAX_SHIFT:g} 之间的有限数字。")
    return number


def normalize_rules(raw: object) -> list[CalibrationRule]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("校准规则须为列表。")
    if len(raw) > MAX_RULES:
        raise ValueError(f"校准规则最多 {MAX_RULES} 条。")
    rules = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != RULE_FIELDS:
            raise ValueError(f"第 {index} 条校准规则须包含且仅包含 meter、starts、ends、lower、upper 字段。")
        starts = _day(item["starts"], RULE_LABELS["starts"])
        ends = _day(item["ends"], RULE_LABELS["ends"])
        if starts > ends:
            raise ValueError(f"第 {index} 条校准规则的测读时段起止颠倒。")
        lower = _shift(item["lower"], RULE_LABELS["lower"])
        upper = _shift(item["upper"], RULE_LABELS["upper"])
        if lower > upper:
            raise ValueError(f"第 {index} 条校准规则的校正下限不得高于上限。")
        rules.append(CalibrationRule(meter=_meter(item["meter"]), starts=starts, ends=ends, lower=lower, upper=upper))
    rules.sort(key=lambda rule: (rule["starts"], rule["ends"], rule["meter"]))
    for previous, current in zip(rules, rules[1:]):
        if current["starts"] <= previous["ends"]:
            raise ValueError(
                f"测读时段重叠：{previous['meter']}（{previous['starts']} 至 {previous['ends']}）与 "
                f"{current['meter']}（{current['starts']} 至 {current['ends']}）。同一时段只启用一个仪表的校准规则。"
            )
    return rules


def rule_for(rules: list[CalibrationRule], day: str) -> CalibrationRule | None:
    for rule in rules:
        if rule["starts"] <= day <= rule["ends"]:
            return rule
    return None


def _clamp(value: float) -> float:
    low, high = BOUNDS["humidity_pct"]
    return min(high, max(low, value))


def correct_rows(rows: list[Row], rules: list[CalibrationRule], bound: Literal["lower", "upper"]) -> list[Row]:
    corrected = []
    for row in rows:
        rule = rule_for(rules, row["work_date"])
        if rule is None:
            corrected.append(row)
        else:
            corrected.append({**row, "humidity_pct": _clamp(row["humidity_pct"] + rule[bound])})
    return corrected


def _correction(row: Row, rules: list[CalibrationRule]) -> tuple[float, float] | None:
    rule = rule_for(rules, row["work_date"])
    if rule is None:
        return None
    return (_clamp(row["humidity_pct"] + rule["lower"]), _clamp(row["humidity_pct"] + rule["upper"]))


def affected_orders(rows: list[Row], old_rules: list[CalibrationRule], new_rules: list[CalibrationRule]) -> tuple[bool, list[str]]:
    """比较两版规则对台账的实际更正：返回（训练读数是否变化，估时须暂停的待排班工单号）。"""
    training = False
    paused = []
    for row in rows:
        if _correction(row, old_rules) != _correction(row, new_rules):
            if row["record_type"] == "history":
                training = True
            else:
                paused.append(row["id"])
    return training, sorted(paused)
