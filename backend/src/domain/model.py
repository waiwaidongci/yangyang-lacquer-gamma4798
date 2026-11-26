"""Forward holdout and Gamma GLM using only the retained training records."""

from dataclasses import dataclass
from math import ceil, fsum
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import StandardScaler

from .records import FEATURES, Row


@dataclass(frozen=True)
class Partition:
    train: list[Row]
    holdout: list[Row]
    excluded: list[Row]
    planned: list[Row]


def chronological_split(rows: list[Row], holdout_percent: float) -> Partition:
    ordered = sorted(rows, key=lambda row: (row["work_date"], row["id"]))
    history = [row for row in ordered if row["record_type"] == "history"]
    dates = sorted({row["work_date"] for row in history})
    day_count = max(2, ceil(len(dates) * holdout_percent / 100))
    cutoff = dates[-day_count]
    holdout = [row for row in history if row["work_date"] >= cutoff]
    holdout_batches = {row["prep_batch"] for row in holdout}
    train = []
    excluded = []
    for row in history:
        if row["work_date"] < cutoff:
            destination = excluded if row["prep_batch"] in holdout_batches else train
            destination.append(row)
    if len(train) < 16 or len({row["work_date"] for row in train}) < 6:
        raise ValueError("配漆批次隔离后，训练集须至少保留 16 条工单和 6 个作业日。请补齐较早历史或减少留出比例。")
    if len(holdout) < 4:
        raise ValueError("留出日期须至少覆盖 4 条已完工工单。请补齐末期历史或提高留出比例。")
    return Partition(train, holdout, excluded, [row for row in ordered if row["record_type"] == "planned"])


def feature_matrix(rows: list[Row]) -> np.ndarray:
    return np.asarray([[row[key] for key in FEATURES] for row in rows], dtype=float)


@dataclass(frozen=True)
class Fit:
    scaler: StandardScaler
    model: GammaRegressor
    train_mean: float
    feature_min: np.ndarray
    feature_max: np.ndarray

    def predict(self, rows: list[Row]) -> list[float]:
        if not rows:
            return []
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                predictions = self.model.predict(self.scaler.transform(feature_matrix(rows)))
        except FloatingPointError as error:
            raise ValueError("预测超出计算范围，请检查工艺条件的跨度及训练数据。") from error
        if not np.all(np.isfinite(predictions) & (predictions > 0)):
            raise ValueError("模型未能生成有效的正值工时，请检查工艺条件的跨度及训练数据。")
        return predictions.tolist()


def fit_gamma(train: list[Row], alpha: float) -> Fit:
    features = feature_matrix(train)
    targets = [row["dry_hours"] for row in train]
    scaler = StandardScaler().fit(features)
    model = GammaRegressor(alpha=alpha, solver="lbfgs", max_iter=1_000, tol=1e-9)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(scaler.transform(features), targets)
    except ConvergenceWarning as error:
        raise ValueError("Gamma 拟合未收敛。请增大正则系数，或检查历史工时与工艺记录。") from error
    return Fit(scaler, model, fsum(targets) / len(targets), features.min(axis=0), features.max(axis=0))
