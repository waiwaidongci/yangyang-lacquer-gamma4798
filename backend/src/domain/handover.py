"""交接闸门：校准版本不符时，估时表的失效与待排班工单暂停视图。"""

from .diagnostics import planned_batch_summary


# 由模型拟合派生、校准更新后即失效的结果表。
ESTIMATE_TABLES = frozenset({
    "holdout_comparison", "planned_orders", "planned_batches",
    "holdout_orders", "model_effects", "calibration_orders",
})
# 只更正待排班读数时，需要按工单暂停估时的结果表。
HANDOVER_TABLES = frozenset({"planned_orders", "planned_batches", "calibration_orders"})
PAUSED_STATUS = "暂停交接（校准已更新）"


def handover_table(tables: list[dict], key: str, paused: set[str]) -> dict:
    """暂停工单的估时不导出：估时字段置空并标记，批次汇总只统计照常交接的工单。"""
    table = next(table for table in tables if table["key"] == key)
    if key == "planned_batches":
        planned = next(table for table in tables if table["key"] == "planned_orders")
        cleared = [row for row in planned["rows"] if row["id"] not in paused]
        return {**table, "rows": planned_batch_summary(cleared)}
    rows = []
    for row in table["rows"]:
        if row["id"] not in paused:
            rows.append(row)
        elif key == "planned_orders":
            rows.append({**row, "estimated_hours": None, "gap_hours": None, "plan_status": PAUSED_STATUS})
        else:
            rows.append({
                **row, "lower_estimate_hours": None, "upper_estimate_hours": None,
                "adopted_hours": None, "adopted_bound": "暂停交接",
            })
    return {**table, "rows": rows}
