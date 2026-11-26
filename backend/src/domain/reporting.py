"""Stable result keys and field labels shared by the UI and CSV export."""

from math import exp

from .calibration import BOUNDS_ORDER, correct_rows, rule_for
from .diagnostics import day_audit, holdout_scores, planned_batch_summary, planned_comparisons
from .model import Fit, Partition, fit_gamma
from .records import FEATURES, FIELDS


ORDER_LABELS = {key: FIELDS[key] for key in ("id", "work_date", "prep_batch")}
ESTIMATE_LABEL = "Gamma 估时（小时）"
SCENARIO_LABELS = {"lower": "下限情景", "upper": "上限情景"}


def _table(key: str, title: str, labels: dict, rows: list[dict]) -> dict:
    return {"key": key, "title": title, "columns": list(labels), "labels": labels, "rows": rows}


def build_report(split: Partition, fitted: Fit) -> dict:
    estimates = fitted.predict(split.holdout)
    scores = holdout_scores(split.holdout, fitted, estimates)
    planning = planned_comparisons(split.planned, fitted)
    over_plan = sum(row["gap_hours"] > 0 for row in planning)
    out_of_range = sum(row["extrapolated_features"] != "无" for row in planning)

    comparisons = [{"method": s.name, "mae_hours": s.mae, "gamma_deviance": s.gamma_deviance} for s in scores]
    residuals = [
        {
            "id": row["id"],
            "work_date": row["work_date"],
            "prep_batch": row["prep_batch"],
            "dry_hours": row["dry_hours"],
            "estimated_hours": estimate,
            "residual_hours": row["dry_hours"] - estimate,
            "planned_hours": row["planned_hours"],
        }
        for row, estimate in zip(split.holdout, estimates, strict=True)
    ]
    coefficients = [
        {
            "feature": FIELDS[key],
            "log_coefficient": float(fitted.model.coef_[i]),
            "mean_multiplier": exp(float(fitted.model.coef_[i])),
            "training_sd": float(fitted.scaler.var_[i] ** 0.5),
            "training_min": float(fitted.feature_min[i]),
            "training_max": float(fitted.feature_max[i]),
        }
        for i, key in enumerate(FEATURES)
    ]
    excluded_orders = [{key: row[key] for key in ORDER_LABELS} for row in split.excluded]
    tables = [
        _table("holdout_comparison", "留出表现与排班基准", {
            "method": "对照口径", "mae_hours": "平均绝对误差（小时）", "gamma_deviance": "平均 Gamma 偏差",
        }, comparisons),
        _table("planned_orders", "待排班工单估时对照", {
            **ORDER_LABELS, "planned_hours": FIELDS["planned_hours"], "estimated_hours": ESTIMATE_LABEL,
            "gap_hours": "估时减预留（小时）", "plan_status": "排班标记", "extrapolated_features": "超训练范围的条件",
        }, planning),
        _table("planned_batches", "待排班配漆批次汇总", {
            "work_date": FIELDS["work_date"], "prep_batch": FIELDS["prep_batch"], "orders": "工单数",
            "over_plan": "超预留工单数", "largest_gap_hours": "最大工单缺口（小时）", "first_order": "最大缺口对应工单",
        }, planned_batch_summary(planning)),
        _table("holdout_orders", "留出工单残差", {
            **ORDER_LABELS, "dry_hours": FIELDS["dry_hours"], "estimated_hours": ESTIMATE_LABEL,
            "residual_hours": "实测减估时（小时）", "planned_hours": FIELDS["planned_hours"],
        }, residuals),
        _table("day_split", "作业日切分台账", {
            "work_date": FIELDS["work_date"], "training": "训练工单", "holdout": "留出工单",
            "excluded": "跨界隔离工单", "planned": "待排班工单",
        }, day_audit(split)),
        _table("batch_exclusions", "跨界配漆批次隔离明细", ORDER_LABELS, excluded_orders),
        _table("model_effects", "模型系数与训练条件范围", {
            "feature": "工艺条件", "log_coefficient": "标准化对数系数", "mean_multiplier": "每增一个训练标准差的均值倍数",
            "training_sd": "训练标准差", "training_min": "训练最小值", "training_max": "训练最大值",
        }, coefficients),
    ]
    warnings = []
    if split.excluded:
        batch_count = len({row["prep_batch"] for row in split.excluded})
        warnings.append(f"{batch_count} 个配漆批次跨过切点，较早一侧的 {len(split.excluded)} 条工单已隔离。")
    if out_of_range:
        warnings.append(f"{out_of_range} 条待排班工单超出训练中至少一项工艺范围，发放排班前请核对测读条件。")
    if not split.planned:
        warnings.append("本次来源没有待排班工单，排班对照表为空。")
    if any(fitted.scaler.var_ == 0):
        warnings.append("训练期有工艺条件保持恒定，本次数据无法估计该条件变化带来的影响。")

    return {
        "summary": (
            f"用 {len(split.train)} 条较早完工工单拟合对数链接 Gamma 回归。"
            f"{split.holdout[0]['work_date']} 至 {split.holdout[-1]['work_date']} 的 {len(split.holdout)} 条留出工单"
            f"平均绝对误差为 {scores[0].mae:.2f} 小时。{len(planning)} 条待排班工单中有 {over_plan} 条估时超过预留。"
        ),
        "metrics": {
            "训练工单数": len(split.train),
            "留出工单数": len(split.holdout),
            "留出平均绝对误差（小时）": round(scores[0].mae, 3),
            "留出平均 Gamma 偏差": round(scores[0].gamma_deviance, 6),
            "超预留计划数": over_plan,
        },
        "tables": tables,
        "warnings": warnings,
    }


def _corrected_partition(split: Partition, rules: list, bound: str) -> Partition:
    return Partition(
        correct_rows(split.train, rules, bound),
        correct_rows(split.holdout, rules, bound),
        split.excluded,
        correct_rows(split.planned, rules, bound),
    )


def _merge_plans(lower_plan: list[dict], upper_plan: list[dict]) -> list[dict]:
    """同一工单在两个校准情景下的估时取较长者，越界提示取两情景的并集。"""
    upper_by_id = {row["id"]: row for row in upper_plan}
    labels = [FIELDS[key] for key in FEATURES]
    merged = []
    for lower in lower_plan:
        upper = upper_by_id[lower["id"]]
        estimate = max(lower["estimated_hours"], upper["estimated_hours"])
        gap = round(estimate - lower["planned_hours"], 2) or 0.0
        flagged = {*lower["extrapolated_features"].split("、"), *upper["extrapolated_features"].split("、")} - {"无"}
        merged.append({
            "id": lower["id"],
            "work_date": lower["work_date"],
            "prep_batch": lower["prep_batch"],
            "planned_hours": lower["planned_hours"],
            "estimated_hours": estimate,
            "gap_hours": gap,
            "plan_status": "复核预留" if gap > 0 else "预留覆盖估时",
            "extrapolated_features": "、".join(label for label in labels if label in flagged) or "无",
        })
    return sorted(merged, key=lambda row: (-row["gap_hours"], row["work_date"], row["id"]))


def build_calibrated_report(split: Partition, rules: list, alpha: float) -> dict:
    """按校准偏差的上下限各做一次拟合与估时，每件工单采用两次中较长的工时。"""
    corrected = {bound: _corrected_partition(split, rules, bound) for bound in BOUNDS_ORDER}
    fits = {bound: fit_gamma(corrected[bound].train, alpha) for bound in BOUNDS_ORDER}
    estimates = [
        max(lower, upper)
        for lower, upper in zip(
            fits["lower"].predict(corrected["lower"].holdout),
            fits["upper"].predict(corrected["upper"].holdout),
            strict=True,
        )
    ]
    scores = holdout_scores(split.holdout, fits["lower"], estimates)
    plans = {bound: planned_comparisons(corrected[bound].planned, fits[bound]) for bound in BOUNDS_ORDER}
    planning = _merge_plans(plans["lower"], plans["upper"])
    over_plan = sum(row["gap_hours"] > 0 for row in planning)
    out_of_range = sum(row["extrapolated_features"] != "无" for row in planning)

    comparisons = [{"method": s.name, "mae_hours": s.mae, "gamma_deviance": s.gamma_deviance} for s in scores]
    residuals = [
        {
            "id": row["id"],
            "work_date": row["work_date"],
            "prep_batch": row["prep_batch"],
            "dry_hours": row["dry_hours"],
            "estimated_hours": estimate,
            "residual_hours": row["dry_hours"] - estimate,
            "planned_hours": row["planned_hours"],
        }
        for row, estimate in zip(split.holdout, estimates, strict=True)
    ]
    coefficients = [
        {
            "scenario": SCENARIO_LABELS[bound],
            "feature": FIELDS[key],
            "log_coefficient": float(fits[bound].model.coef_[i]),
            "mean_multiplier": exp(float(fits[bound].model.coef_[i])),
            "training_sd": float(fits[bound].scaler.var_[i] ** 0.5),
            "training_min": float(fits[bound].feature_min[i]),
            "training_max": float(fits[bound].feature_max[i]),
        }
        for bound in BOUNDS_ORDER
        for i, key in enumerate(FEATURES)
    ]
    scenario_estimates = {
        bound: {row["id"]: row["estimated_hours"] for row in plans[bound]} for bound in BOUNDS_ORDER
    }
    scenario_orders = [
        {
            "id": record["id"],
            "work_date": record["work_date"],
            "prep_batch": record["prep_batch"],
            "lower_estimate_hours": scenario_estimates["lower"][record["id"]],
            "upper_estimate_hours": scenario_estimates["upper"][record["id"]],
            "adopted_hours": max(scenario_estimates["lower"][record["id"]], scenario_estimates["upper"][record["id"]]),
            "adopted_bound": "上限" if scenario_estimates["upper"][record["id"]] > scenario_estimates["lower"][record["id"]] else "下限",
        }
        for record in split.planned
    ]
    excluded_orders = [{key: row[key] for key in ORDER_LABELS} for row in split.excluded]
    tables = [
        _table("holdout_comparison", "留出表现与排班基准", {
            "method": "对照口径", "mae_hours": "平均绝对误差（小时）", "gamma_deviance": "平均 Gamma 偏差",
        }, comparisons),
        _table("planned_orders", "待排班工单估时对照", {
            **ORDER_LABELS, "planned_hours": FIELDS["planned_hours"], "estimated_hours": ESTIMATE_LABEL,
            "gap_hours": "估时减预留（小时）", "plan_status": "排班标记", "extrapolated_features": "超训练范围的条件",
        }, planning),
        _table("planned_batches", "待排班配漆批次汇总", {
            "work_date": FIELDS["work_date"], "prep_batch": FIELDS["prep_batch"], "orders": "工单数",
            "over_plan": "超预留工单数", "largest_gap_hours": "最大工单缺口（小时）", "first_order": "最大缺口对应工单",
        }, planned_batch_summary(planning)),
        _table("calibration_orders", "待排班工单分情景估时", {
            **ORDER_LABELS, "lower_estimate_hours": "下限情景估时（小时）", "upper_estimate_hours": "上限情景估时（小时）",
            "adopted_hours": "采用估时（小时）", "adopted_bound": "采用情景",
        }, scenario_orders),
        _table("holdout_orders", "留出工单残差", {
            **ORDER_LABELS, "dry_hours": FIELDS["dry_hours"], "estimated_hours": ESTIMATE_LABEL,
            "residual_hours": "实测减估时（小时）", "planned_hours": FIELDS["planned_hours"],
        }, residuals),
        _table("day_split", "作业日切分台账", {
            "work_date": FIELDS["work_date"], "training": "训练工单", "holdout": "留出工单",
            "excluded": "跨界隔离工单", "planned": "待排班工单",
        }, day_audit(split)),
        _table("batch_exclusions", "跨界配漆批次隔离明细", ORDER_LABELS, excluded_orders),
        _table("model_effects", "模型系数与训练条件范围", {
            "scenario": "校准情景", "feature": "工艺条件", "log_coefficient": "标准化对数系数",
            "mean_multiplier": "每增一个训练标准差的均值倍数", "training_sd": "训练标准差",
            "training_min": "训练最小值", "training_max": "训练最大值",
        }, coefficients),
    ]
    corrected_count = sum(
        1
        for group in (split.train, split.holdout, split.excluded, split.planned)
        for row in group
        if rule_for(rules, row["work_date"]) is not None
    )
    warnings = []
    if split.excluded:
        batch_count = len({row["prep_batch"] for row in split.excluded})
        warnings.append(f"{batch_count} 个配漆批次跨过切点，较早一侧的 {len(split.excluded)} 条工单已隔离。")
    if out_of_range:
        warnings.append(f"{out_of_range} 条待排班工单超出训练中至少一项工艺范围，发放排班前请核对测读条件。")
    if not split.planned:
        warnings.append("本次来源没有待排班工单，排班对照表为空。")
    if any((fits[bound].scaler.var_ == 0).any() for bound in BOUNDS_ORDER):
        warnings.append("训练期有工艺条件保持恒定，本次数据无法估计该条件变化带来的影响。")
    warnings.append(f"湿度校准已生效：{len(rules)} 条规则更正 {corrected_count} 条工单读数，估时取上下限情景中较长者。")

    return {
        "summary": (
            f"用 {len(split.train)} 条较早完工工单拟合对数链接 Gamma 回归。"
            f"{split.holdout[0]['work_date']} 至 {split.holdout[-1]['work_date']} 的 {len(split.holdout)} 条留出工单"
            f"平均绝对误差为 {scores[0].mae:.2f} 小时。{len(planning)} 条待排班工单中有 {over_plan} 条估时超过预留。"
            f"已按 {len(rules)} 条仪表校准规则更正湿度，上下限情景各拟合一次并取较长估时。"
        ),
        "metrics": {
            "训练工单数": len(split.train),
            "留出工单数": len(split.holdout),
            "留出平均绝对误差（小时）": round(scores[0].mae, 3),
            "留出平均 Gamma 偏差": round(scores[0].gamma_deviance, 6),
            "超预留计划数": over_plan,
            "校准规则数": len(rules),
            "受更正工单数": corrected_count,
        },
        "tables": tables,
        "warnings": warnings,
        "calibration": {"rules": rules, "corrected_orders": corrected_count},
    }
