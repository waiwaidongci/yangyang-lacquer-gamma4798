"""CSV contract and validation for closed and scheduled lacquer work orders."""

from datetime import date
from math import isfinite
import re
from typing import Literal, TypedDict


class Row(TypedDict):
    id: str
    work_date: str
    prep_batch: str
    record_type: Literal["history", "planned"]
    film_um: float
    temperature_c: float
    humidity_pct: float
    planned_hours: float
    dry_hours: float | None


FIELDS = {
    "id": "雕版工单号",
    "work_date": "补漆作业日期",
    "prep_batch": "配漆批次",
    "record_type": "记录类型",
    "film_um": "膜厚（µm）",
    "temperature_c": "温度（℃）",
    "humidity_pct": "相对湿度（%）",
    "planned_hours": "排班预留（小时）",
    "dry_hours": "干燥实测（小时）",
}
FEATURES = ("film_um", "temperature_c", "humidity_pct")
BOUNDS = {
    "film_um": (10, 250),
    "temperature_c": (5, 40),
    "humidity_pct": (15, 95),
    "planned_hours": (0.01, 240),
    "dry_hours": (0.01, 240),
}
_IDENTIFIER = re.compile(r"[^\W_][\w.-]{0,63}")


def _identifier(value: object, label: str) -> str:
    if isinstance(value, str) and _IDENTIFIER.fullmatch(value.strip()):
        return value.strip()
    raise ValueError(f"{label}须以字母或数字开头，且只可使用字母、数字、下划线、点或连字符，最多 64 字符。")


def _work_date(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        try:
            if date.fromisoformat(text).isoformat() == text:
                return text
        except ValueError:
            pass
    raise ValueError("补漆作业日期须为有效的 YYYY-MM-DD 日期。")


def _number(value: object, key: str) -> float:
    low, high = BOUNDS[key]
    try:
        valid_type = isinstance(value, (str, int, float)) and not isinstance(value, bool)
        number = float(value) if valid_type else float("nan")
    except (ValueError, OverflowError):
        number = float("nan")
    if not isfinite(number) or not low <= number <= high:
        raise ValueError(f"{FIELDS[key]}须为 {low:g} 至 {high:g} 之间的有限数字。")
    return number


def _read(row: dict) -> Row:
    kind = row["record_type"].strip() if isinstance(row["record_type"], str) else None
    if kind not in ("history", "planned"):
        raise ValueError("记录类型须为 history（已完工）或 planned（待排班）。")
    actual = row["dry_hours"]
    if kind == "planned" and actual is not None and (not isinstance(actual, str) or actual.strip()):
        raise ValueError("待排班记录的干燥实测须留空。")
    return Row(
        id=_identifier(row["id"], FIELDS["id"]),
        work_date=_work_date(row["work_date"]),
        prep_batch=_identifier(row["prep_batch"], FIELDS["prep_batch"]),
        record_type=kind,
        film_um=_number(row["film_um"], "film_um"),
        temperature_c=_number(row["temperature_c"], "temperature_c"),
        humidity_pct=_number(row["humidity_pct"], "humidity_pct"),
        planned_hours=_number(row["planned_hours"], "planned_hours"),
        dry_hours=_number(actual, "dry_hours") if kind == "history" else None,
    )


def normalize_rows(rows: list[dict]) -> list[Row]:
    normalized = []
    seen = set()
    for line, row in enumerate(rows, start=2):
        if not isinstance(row, dict):
            raise ValueError(f"第 {line} 行须为一条工单记录。")
        if row.keys() != FIELDS.keys():
            missing = "、".join(sorted(FIELDS.keys() - row.keys())) or "无"
            extra = "、".join(sorted(str(key) for key in row.keys() - FIELDS.keys())) or "无"
            raise ValueError(f"第 {line} 行字段不符（缺少 {missing}，多出 {extra}）。")
        try:
            record = _read(row)
        except ValueError as error:
            raise ValueError(f"第 {line} 行：{error}") from error
        if record["id"] in seen:
            raise ValueError(f"第 {line} 行工单号重复：{record['id']}。")
        seen.add(record["id"])
        normalized.append(record)

    history = [row for row in normalized if row["record_type"] == "history"]
    dates = {row["work_date"] for row in history}
    if len(history) < 24 or len(dates) < 10:
        raise ValueError("至少需要 24 条已完工记录，且须覆盖至少 10 个作业日。")
    last_history_date = max(dates)
    for row in normalized:
        if row["record_type"] == "planned" and row["work_date"] <= last_history_date:
            raise ValueError(f"待排班工单 {row['id']} 的日期须晚于末个已完工作业日 {last_history_date}。")
    return normalized
