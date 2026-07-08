from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.common import (
    DEFAULT_EXCEL_PATH,
    aggregate_weekly_sales,
    clean_sales_frame,
    load_sales_excel,
    make_postgres_engine,
    table_reference,
)
from src.db_schema import prepare_table
from src.export_predictions import export_predictions_to_excel


REPEAT_HORIZON_DAYS = 90
PASSIVE_PRODUCT_DAYS = 90
ACTION_MIN_REPEAT_PROBABILITY = 0.50
ACTION_MIN_CONFIDENCE_SCORE = 60

STATUS_VALID = "Ge\u00e7erli Tahmin"
STATUS_OVERDUE = "Gecikmi\u015f / Overdue"
STATUS_PASSIVE = "Pasif \u00dcr\u00fcn"
STATUS_INSUFFICIENT = "Yetersiz Ge\u00e7mi\u015f"
STATUS_LOW_PROBABILITY = "D\u00fc\u015f\u00fck Olas\u0131l\u0131kl\u0131 Tahmin"

CONFIDENCE_HIGH = "Y\u00fcksek G\u00fcven"
CONFIDENCE_MEDIUM = "Orta G\u00fcven"
CONFIDENCE_LOW = "D\u00fc\u015f\u00fck G\u00fcven / Yetersiz Ge\u00e7mi\u015f"

STATUS_PRIORITY = {
    STATUS_VALID: 0,
    STATUS_OVERDUE: 1,
    STATUS_LOW_PROBABILITY: 2,
    STATUS_INSUFFICIENT: 3,
    STATUS_PASSIVE: 4,
}

FEATURE_COLUMNS = [
    "cari_id",
    "stock_id",
    "toplam_miktar",
    "toplam_tutar",
    "siparis_satir_sayisi",
    "unit_price",
    "weekofyear",
    "month",
    "quarter",
    "year",
    "week_sin",
    "week_cos",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "order_count_so_far",
    "days_since_first_order",
    "last_interval_days",
    "avg_interval_days",
    "median_interval_days",
    "interval_std_days",
    "interval_cv",
    "interval_min_days",
    "interval_max_days",
    "qty_mean_so_far",
    "qty_median_so_far",
    "qty_last3_mean_so_far",
    "qty_last6_mean_so_far",
    "amount_mean_so_far",
    "line_mean_so_far",
    "qty_sum_so_far",
    "amount_sum_so_far",
    "qty_std_so_far",
    "pair_active_month_count",
    "pair_dominant_month_ratio",
    "pair_dominant_quarter_ratio",
    "pair_current_month_ratio",
    "pair_ilkbahar_share",
    "pair_yaz_share",
    "pair_sonbahar_share",
    "pair_kis_share",
]

PAIR_MONTH_COUNT_COLUMNS = [f"pair_month_{month}_count" for month in range(1, 13)]
PAIR_QUARTER_COUNT_COLUMNS = [f"pair_quarter_{quarter}_count" for quarter in range(1, 5)]
SEASON_MONTHS = {
    "İlkbahar": [3, 4, 5],
    "Yaz": [6, 7, 8],
    "Sonbahar": [9, 10, 11],
    "Kış": [12, 1, 2],
}
SEASON_COLUMN_SUFFIX = {
    "İlkbahar": "ilkbahar",
    "Yaz": "yaz",
    "Sonbahar": "sonbahar",
    "Kış": "kis",
}
MONTH_NAMES = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-order models and export predictions.")
    parser.add_argument("--source", choices=["excel", "postgres"], default="excel")
    parser.add_argument("--excel", default=str(DEFAULT_EXCEL_PATH), help="Excel path used when --source excel.")
    parser.add_argument("--table", default="historical_sales_weekly", help="PostgreSQL source table.")
    parser.add_argument("--schema", default=None, help="Optional PostgreSQL schema.")
    parser.add_argument("--output", default="outputs/next_order_predictions.xlsx", help="Prediction Excel path.")
    parser.add_argument("--model-output", default="models/next_order_models.joblib", help="Model artifact path.")
    parser.add_argument("--predictions-table", default="next_order_predictions", help="PostgreSQL prediction table.")
    parser.add_argument("--write-postgres", action="store_true", help="Also write predictions to PostgreSQL.")
    parser.add_argument("--min-training-orders", type=int, default=2, help="Minimum orders needed for training rows.")
    parser.add_argument(
        "--repeat-horizon-days",
        type=int,
        default=REPEAT_HORIZON_DAYS,
        help="Classifier target horizon for repeat purchase.",
    )
    return parser.parse_args()


def read_sales_from_postgres(table: str, schema: str | None) -> pd.DataFrame:
    engine = make_postgres_engine()
    query = f"SELECT * FROM {table_reference(table, schema)}"
    frame = pd.read_sql_query(query, engine)
    return clean_sales_frame(frame)


def _reset_pair_index(series: pd.Series) -> pd.Series:
    return series.reset_index(level=[0, 1], drop=True)


def add_time_series_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_weekly_sales(frame)
    if df.empty:
        raise ValueError("No positive sales rows were found after cleaning.")

    df = df.sort_values(["cari_id", "stock_id", "hafta"]).reset_index(drop=True)
    pair_keys = ["cari_id", "stock_id"]
    groups = df.groupby(pair_keys, sort=False)

    df["dataset_max_hafta"] = df["hafta"].max()
    df["next_hafta"] = groups["hafta"].shift(-1)
    df["days_until_next"] = (df["next_hafta"] - df["hafta"]).dt.days
    df["next_toplam_miktar"] = groups["toplam_miktar"].shift(-1)
    df["repeat_purchase_90d"] = (
        df["days_until_next"].notna() & (df["days_until_next"] <= REPEAT_HORIZON_DAYS)
    ).astype("int64")

    df["first_hafta"] = groups["hafta"].transform("first")
    df["order_count_so_far"] = groups.cumcount() + 1
    df["days_since_first_order"] = (df["hafta"] - df["first_hafta"]).dt.days
    df["last_interval_days"] = groups["hafta"].diff().dt.days
    df["interval_count_so_far"] = df["last_interval_days"].notna().astype("int64")
    df["interval_count_so_far"] = df.groupby(pair_keys, sort=False)["interval_count_so_far"].cumsum()

    interval_for_sum = df["last_interval_days"].fillna(0)
    df["interval_sum_so_far"] = interval_for_sum.groupby([df["cari_id"], df["stock_id"]], sort=False).cumsum()
    df["interval_sq_sum_so_far"] = (interval_for_sum**2).groupby(
        [df["cari_id"], df["stock_id"]], sort=False
    ).cumsum()
    df["avg_interval_days"] = df["interval_sum_so_far"] / df["interval_count_so_far"].replace(0, np.nan)
    df["median_interval_days"] = _reset_pair_index(groups["last_interval_days"].expanding().median())
    df["interval_std_days"] = np.sqrt(
        (
            df["interval_sq_sum_so_far"] / df["interval_count_so_far"].replace(0, np.nan)
            - df["avg_interval_days"] ** 2
        ).clip(lower=0)
    )
    df["interval_cv"] = df["interval_std_days"] / df["avg_interval_days"].replace(0, np.nan)
    df["interval_min_days"] = groups["last_interval_days"].cummin()
    df["interval_max_days"] = groups["last_interval_days"].cummax()

    df["qty_sum_so_far"] = groups["toplam_miktar"].cumsum()
    df["amount_sum_so_far"] = groups["toplam_tutar"].cumsum()
    df["line_sum_so_far"] = groups["siparis_satir_sayisi"].cumsum()
    df["qty_mean_so_far"] = df["qty_sum_so_far"] / df["order_count_so_far"]
    df["qty_median_so_far"] = _reset_pair_index(groups["toplam_miktar"].expanding().median())
    df["qty_last3_mean_so_far"] = _reset_pair_index(
        groups["toplam_miktar"].rolling(window=3, min_periods=1).mean()
    )
    df["qty_last6_mean_so_far"] = _reset_pair_index(
        groups["toplam_miktar"].rolling(window=6, min_periods=1).mean()
    )
    df["qty_p95_so_far"] = _reset_pair_index(groups["toplam_miktar"].expanding().quantile(0.95))
    df["amount_mean_so_far"] = df["amount_sum_so_far"] / df["order_count_so_far"]
    df["line_mean_so_far"] = df["line_sum_so_far"] / df["order_count_so_far"]
    df["qty_sq_sum_so_far"] = (df["toplam_miktar"] ** 2).groupby(
        [df["cari_id"], df["stock_id"]], sort=False
    ).cumsum()
    df["qty_std_so_far"] = np.sqrt(
        (df["qty_sq_sum_so_far"] / df["order_count_so_far"] - df["qty_mean_so_far"] ** 2).clip(lower=0)
    )
    df["unit_price"] = np.where(df["toplam_miktar"] > 0, df["toplam_tutar"] / df["toplam_miktar"], 0)

    iso_calendar = df["hafta"].dt.isocalendar()
    df["weekofyear"] = iso_calendar.week.astype("int64")
    df["month"] = df["hafta"].dt.month
    df["quarter"] = df["hafta"].dt.quarter
    df["year"] = df["hafta"].dt.year
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 53)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 53)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

    for month in range(1, 13):
        column = f"pair_month_{month}_count"
        df[column] = (df["month"] == month).astype("int64")
        df[column] = df.groupby(pair_keys, sort=False)[column].cumsum()

    for quarter in range(1, 5):
        column = f"pair_quarter_{quarter}_count"
        df[column] = (df["quarter"] == quarter).astype("int64")
        df[column] = df.groupby(pair_keys, sort=False)[column].cumsum()

    month_counts = df[PAIR_MONTH_COUNT_COLUMNS]
    quarter_counts = df[PAIR_QUARTER_COUNT_COLUMNS]
    df["pair_active_month_count"] = (month_counts > 0).sum(axis=1)
    df["pair_dominant_month_ratio"] = month_counts.max(axis=1) / df["order_count_so_far"]
    df["pair_dominant_quarter_ratio"] = quarter_counts.max(axis=1) / df["order_count_so_far"]
    df["pair_current_month_ratio"] = 0.0
    for month in range(1, 13):
        month_mask = df["month"] == month
        df.loc[month_mask, "pair_current_month_ratio"] = (
            df.loc[month_mask, f"pair_month_{month}_count"] / df.loc[month_mask, "order_count_so_far"]
        )

    for season, months in SEASON_MONTHS.items():
        count_columns = [f"pair_month_{month}_count" for month in months]
        suffix = SEASON_COLUMN_SUFFIX[season]
        df[f"pair_{suffix}_share"] = df[count_columns].sum(axis=1) / df["order_count_so_far"]

    global_interval_median = df["last_interval_days"].median()
    if pd.isna(global_interval_median):
        global_interval_median = 28
    df["last_interval_days"] = df["last_interval_days"].fillna(global_interval_median)
    df["avg_interval_days"] = df["avg_interval_days"].fillna(df["last_interval_days"])
    df["median_interval_days"] = df["median_interval_days"].fillna(df["avg_interval_days"])
    df["interval_std_days"] = df["interval_std_days"].fillna(0)
    df["interval_cv"] = df["interval_cv"].fillna(0)
    df["interval_min_days"] = df["interval_min_days"].fillna(df["last_interval_days"])
    df["interval_max_days"] = df["interval_max_days"].fillna(df["last_interval_days"])
    df["son_siparisten_gecen_gun"] = (df["dataset_max_hafta"] - df["hafta"]).dt.days
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_models() -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor, TransformedTargetRegressor]:
    repeat_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        l2_regularization=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        random_state=41,
    )
    days_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=320,
        learning_rate=0.045,
        l2_regularization=0.05,
        max_leaf_nodes=47,
        min_samples_leaf=18,
        random_state=42,
    )
    quantity_model = TransformedTargetRegressor(
        regressor=HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            l2_regularization=0.05,
            min_samples_leaf=25,
            random_state=43,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return repeat_model, days_model, quantity_model


def round_to_week_days(values: Any) -> np.ndarray:
    return np.maximum(7, np.round(np.asarray(values, dtype=float) / 7) * 7).astype("int64")


def confidence_level_from_count(order_count: pd.Series | np.ndarray) -> pd.Series:
    counts = pd.Series(order_count)
    levels = pd.Series(CONFIDENCE_LOW, index=counts.index, dtype="object")
    levels.loc[counts.between(3, 5)] = CONFIDENCE_MEDIUM
    levels.loc[counts >= 6] = CONFIDENCE_HIGH
    return levels


def smape(actual: Any, predicted: Any) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    valid = denominator > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(2 * np.abs(predicted_array[valid] - actual_array[valid]) / denominator[valid]) * 100)


def safe_r2(actual: Any, predicted: Any) -> float:
    if len(actual) < 2:
        return float("nan")
    try:
        return float(r2_score(actual, predicted))
    except ValueError:
        return float("nan")


def time_based_split(frame: pd.DataFrame, date_column: str = "hafta") -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    frame = frame.sort_values(date_column).reset_index(drop=True)
    unique_dates = pd.Series(frame[date_column].dropna().sort_values().unique())
    if len(unique_dates) < 2:
        split_index = min(max(int(len(frame) * 0.8), 1), len(frame) - 1)
        split_date = pd.Timestamp(frame.iloc[split_index][date_column])
        return frame.iloc[:split_index], frame.iloc[split_index:], split_date

    split_position = min(max(int(len(unique_dates) * 0.8), 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates.iloc[split_position])
    train = frame[frame[date_column] < split_date].copy()
    valid = frame[frame[date_column] >= split_date].copy()
    if train.empty or valid.empty:
        split_index = min(max(int(len(frame) * 0.8), 1), len(frame) - 1)
        train = frame.iloc[:split_index].copy()
        valid = frame.iloc[split_index:].copy()
        split_date = pd.Timestamp(valid[date_column].min())
    return train, valid, split_date


def describe_pair_pattern(row: pd.Series) -> tuple[str, str, str]:
    avg_interval = float(row["avg_interval_days"])
    interval_cv = float(row["interval_cv"])
    interval_count = int(row["interval_count_so_far"])
    order_count = int(row["order_count_so_far"])
    active_month_count = int(row["pair_active_month_count"])
    dominant_month_ratio = float(row["pair_dominant_month_ratio"])
    dominant_quarter_ratio = float(row["pair_dominant_quarter_ratio"])
    season_shares = {
        season: float(row[f"pair_{suffix}_share"])
        for season, suffix in SEASON_COLUMN_SUFFIX.items()
    }
    dominant_season, dominant_season_share = max(season_shares.items(), key=lambda item: item[1])

    active_months = [
        MONTH_NAMES[month]
        for month, column in enumerate(PAIR_MONTH_COUNT_COLUMNS, start=1)
        if int(row[column]) > 0
    ]
    active_months_text = ", ".join(active_months) if active_months else "Yetersiz veri"

    if order_count >= 4 and avg_interval <= 10 and interval_cv <= 0.45:
        pattern = "Haftalık"
    elif order_count >= 4 and 11 <= avg_interval <= 20 and interval_cv <= 0.45:
        pattern = "İki haftalık"
    elif order_count >= 4 and 21 <= avg_interval <= 38 and interval_cv <= 0.55:
        pattern = "Aylık"
    elif order_count >= 3 and (active_month_count <= 4 or dominant_quarter_ratio >= 0.70):
        pattern = f"Mevsimsel - {dominant_season}"
    elif order_count >= 3 and (dominant_month_ratio >= 0.45 or dominant_season_share >= 0.60):
        pattern = "Dönemsel"
    else:
        pattern = "Düzensiz / ML ağırlıklı"

    interval_text = (
        f"ortalama aralık {avg_interval:.1f} gün"
        if interval_count >= 1
        else "aralık hesaplanamadı; tek sipariş var"
    )
    reason = (
        f"{pattern}; {order_count} geçmiş sipariş; {interval_text}; en yoğun sezon "
        f"{dominant_season} (%{dominant_season_share * 100:.0f})."
    )
    return pattern, active_months_text, reason


def describe_interval_reliability(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    interval_count = frame["interval_count_so_far"]
    avg_interval = frame["avg_interval_days"].where(interval_count >= 1).round(1)
    regularity_score = (1 / (1 + frame["interval_cv"])).where(interval_count >= 2).round(3)
    interval_note = pd.Series("Yetersiz geçmiş", index=frame.index, dtype="object")
    interval_note.loc[(interval_count >= 2) & (regularity_score >= 0.75)] = "Düzenli"
    interval_note.loc[(interval_count >= 2) & (regularity_score >= 0.45) & (regularity_score < 0.75)] = (
        "Orta düzenli"
    )
    interval_note.loc[(interval_count >= 2) & (regularity_score < 0.45)] = "Düzensiz"
    return avg_interval, regularity_score, interval_note


def predict_quantities(feature_frame: pd.DataFrame, quantity_model: TransformedTargetRegressor) -> np.ndarray:
    raw_quantity = np.clip(quantity_model.predict(feature_frame[FEATURE_COLUMNS]), 0, None)
    rounded_quantity = np.maximum(1, np.floor(raw_quantity + 0.5))

    order_count = feature_frame["order_count_so_far"].to_numpy(dtype=float)
    qty_mean = feature_frame["qty_mean_so_far"].to_numpy(dtype=float)
    qty_median = feature_frame["qty_median_so_far"].to_numpy(dtype=float)
    qty_std = feature_frame["qty_std_so_far"].to_numpy(dtype=float)
    last_qty = feature_frame["toplam_miktar"].to_numpy(dtype=float)

    stable_amount = (order_count >= 5) & (qty_median >= 2) & (qty_std <= np.maximum(qty_mean, 1.0) * 1.25)
    historical_floor = np.floor((0.65 * qty_median) + (0.35 * np.minimum(qty_mean, last_qty)) + 0.5)
    rounded_quantity[stable_amount] = np.maximum(rounded_quantity[stable_amount], historical_floor[stable_amount])

    if "qty_p95_so_far" in feature_frame.columns:
        p95_cap = feature_frame["qty_p95_so_far"].fillna(feature_frame["qty_median_so_far"]).to_numpy(dtype=float)
        p95_cap = np.maximum(1, np.ceil(p95_cap))
        rounded_quantity = np.minimum(rounded_quantity, p95_cap)

    return np.maximum(1, rounded_quantity).astype("int64")


def winsorized_quantity_target(frame: pd.DataFrame) -> pd.Series:
    p95_cap = frame["qty_p95_so_far"].fillna(frame["qty_median_so_far"]).clip(lower=1)
    return frame["next_toplam_miktar"].clip(lower=0).clip(upper=p95_cap)


def baseline_quantity_prediction(frame: pd.DataFrame) -> np.ndarray:
    baseline = frame["qty_median_so_far"].fillna(frame["qty_mean_so_far"]).clip(lower=1)
    if "qty_p95_so_far" in frame.columns:
        baseline = baseline.clip(upper=frame["qty_p95_so_far"].fillna(baseline).clip(lower=1))
    return np.maximum(1, np.floor(baseline.to_numpy(dtype=float) + 0.5)).astype("int64")


def baseline_days_prediction(frame: pd.DataFrame) -> np.ndarray:
    baseline_days = frame["median_interval_days"].fillna(frame["avg_interval_days"]).clip(lower=7)
    return round_to_week_days(baseline_days)


def positive_class_probability(model: Any, feature_frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(feature_frame[FEATURE_COLUMNS])
    classes = getattr(model, "classes_", np.array([0, 1]))
    if 1 in classes:
        positive_index = int(np.where(classes == 1)[0][0])
        return probabilities[:, positive_index]
    return np.zeros(len(feature_frame), dtype=float)


def build_repeat_classifier_training(
    feature_frame: pd.DataFrame,
    min_training_orders: int,
    repeat_horizon_days: int,
) -> pd.DataFrame:
    max_date = feature_frame["hafta"].max()
    complete_window_date = max_date - pd.Timedelta(days=repeat_horizon_days)
    training = feature_frame[
        (feature_frame["order_count_so_far"] >= min_training_orders)
        & (feature_frame["hafta"] <= complete_window_date)
    ].copy()
    training["repeat_purchase_target"] = (
        training["days_until_next"].notna() & (training["days_until_next"] <= repeat_horizon_days)
    ).astype("int64")
    return training


def build_regression_training(feature_frame: pd.DataFrame, min_training_orders: int) -> pd.DataFrame:
    return feature_frame[
        (feature_frame["days_until_next"].notna())
        & (feature_frame["days_until_next"] > 0)
        & (feature_frame["order_count_so_far"] >= min_training_orders)
    ].copy()


def add_validation_predictions(
    valid: pd.DataFrame,
    repeat_model: Any,
    days_model: HistGradientBoostingRegressor,
    quantity_model: TransformedTargetRegressor,
    date_prediction_method: str,
) -> pd.DataFrame:
    result = valid.copy()
    result["repeat_probability_pred"] = positive_class_probability(repeat_model, result)
    result["date_model_days"] = round_to_week_days(np.clip(days_model.predict(result[FEATURE_COLUMNS]), 1, 365))
    result["date_baseline_days"] = baseline_days_prediction(result)
    if date_prediction_method == "median_interval_baseline":
        result["date_predicted_days"] = result["date_baseline_days"]
    else:
        result["date_predicted_days"] = result["date_model_days"]
    result["quantity_predicted"] = predict_quantities(result, quantity_model)
    result["quantity_baseline"] = baseline_quantity_prediction(result)
    result["date_abs_error"] = (result["days_until_next"] - result["date_predicted_days"]).abs()
    result["quantity_abs_error"] = (result["next_toplam_miktar"] - result["quantity_predicted"]).abs()
    result["guven_seviyesi"] = confidence_level_from_count(result["order_count_so_far"].reset_index(drop=True)).values
    return result


def build_error_analysis(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame(
            columns=[
                "GuvenSeviyesi",
                "SatirSayisi",
                "TarihMAE",
                "TarihRMSE",
                "BaselineTarihMAE",
                "TarihIyilesmeYuzde",
                "MiktarMAE",
                "MiktarRMSE",
                "BaselineMiktarMAE",
                "MiktarIyilesmeYuzde",
                "TarihR2",
                "MiktarR2",
                "TarihSMAPE",
                "MiktarSMAPE",
            ]
        )

    rows = []
    for level, group in validation.groupby("guven_seviyesi", dropna=False):
        tarih_mae = float(mean_absolute_error(group["days_until_next"], group["date_predicted_days"]))
        baseline_tarih_mae = float(mean_absolute_error(group["days_until_next"], group["date_baseline_days"]))
        miktar_mae = float(mean_absolute_error(group["next_toplam_miktar"], group["quantity_predicted"]))
        baseline_miktar_mae = float(mean_absolute_error(group["next_toplam_miktar"], group["quantity_baseline"]))
        rows.append(
            {
                "GuvenSeviyesi": level,
                "SatirSayisi": int(len(group)),
                "TarihMAE": tarih_mae,
                "TarihRMSE": float(np.sqrt(mean_squared_error(group["days_until_next"], group["date_predicted_days"]))),
                "BaselineTarihMAE": baseline_tarih_mae,
                "TarihIyilesmeYuzde": (
                    ((baseline_tarih_mae - tarih_mae) / baseline_tarih_mae) * 100
                    if baseline_tarih_mae
                    else float("nan")
                ),
                "MiktarMAE": miktar_mae,
                "MiktarRMSE": float(
                    np.sqrt(mean_squared_error(group["next_toplam_miktar"], group["quantity_predicted"]))
                ),
                "BaselineMiktarMAE": baseline_miktar_mae,
                "MiktarIyilesmeYuzde": (
                    ((baseline_miktar_mae - miktar_mae) / baseline_miktar_mae) * 100
                    if baseline_miktar_mae
                    else float("nan")
                ),
                "TarihR2": safe_r2(group["days_until_next"], group["date_predicted_days"]),
                "MiktarR2": safe_r2(group["next_toplam_miktar"], group["quantity_predicted"]),
                "TarihSMAPE": smape(group["days_until_next"], group["date_predicted_days"]),
                "MiktarSMAPE": smape(group["next_toplam_miktar"], group["quantity_predicted"]),
            }
        )
    return pd.DataFrame(rows).sort_values("GuvenSeviyesi").reset_index(drop=True)


def build_feature_importance(
    repeat_model: Any,
    days_model: HistGradientBoostingRegressor,
    quantity_model: TransformedTargetRegressor,
    classifier_valid: pd.DataFrame,
    regression_valid: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_permutation(model: Any, data: pd.DataFrame, target: pd.Series, model_name: str, scoring: str) -> None:
        if data.empty or target.nunique(dropna=False) < 2:
            return
        sample_size = min(len(data), 600)
        sample = data.sample(n=sample_size, random_state=123) if len(data) > sample_size else data
        target_sample = target.loc[sample.index]
        try:
            importance = permutation_importance(
                model,
                sample[FEATURE_COLUMNS],
                target_sample,
                scoring=scoring,
                n_repeats=2,
                random_state=123,
                n_jobs=1,
            )
        except Exception:
            return
        for feature, mean_score, std_score in zip(
            FEATURE_COLUMNS,
            importance.importances_mean,
            importance.importances_std,
            strict=False,
        ):
            rows.append(
                {
                    "Model": model_name,
                    "Ozellik": feature,
                    "OnemSkoru": float(mean_score),
                    "OnemStd": float(std_score),
                    "Yontem": "permutation_importance",
                }
            )

    if not classifier_valid.empty:
        add_permutation(
            repeat_model,
            classifier_valid,
            classifier_valid["repeat_purchase_target"],
            "Tekrar alma classification",
            "average_precision",
        )
    if not regression_valid.empty:
        add_permutation(
            days_model,
            regression_valid,
            regression_valid["days_until_next"],
            "Sipariş zamanı regression",
            "neg_mean_absolute_error",
        )
        add_permutation(
            quantity_model,
            regression_valid,
            regression_valid["next_toplam_miktar"].clip(lower=0),
            "Sipariş miktarı regression",
            "neg_mean_absolute_error",
        )

    if not rows:
        return pd.DataFrame(columns=["Model", "Ozellik", "OnemSkoru", "OnemStd", "Yontem"])

    importance_frame = pd.DataFrame(rows)
    return (
        importance_frame.sort_values(["Model", "OnemSkoru"], ascending=[True, False])
        .groupby("Model", group_keys=False)
        .head(25)
        .reset_index(drop=True)
    )


def train_models(
    feature_frame: pd.DataFrame,
    min_training_orders: int,
    repeat_horizon_days: int = REPEAT_HORIZON_DAYS,
) -> tuple[Any, HistGradientBoostingRegressor, TransformedTargetRegressor, dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    regression_training = build_regression_training(feature_frame, min_training_orders)
    if len(regression_training) < 100:
        pair_count = feature_frame[["cari_id", "stock_id"]].drop_duplicates().shape[0]
        raise ValueError(
            "Not enough historical repeated orders to train a reliable model. "
            f"Usable positive weekly rows: {len(feature_frame):,}; "
            f"customer-product pairs: {pair_count:,}; "
            f"repeated-order training rows: {len(regression_training):,}. "
            "Check that Hafta, StockId, StockKod, StockAd, ToplamMiktar, ToplamTutar and "
            "SiparisSatirSayisi columns are populated."
        )

    classifier_training = build_repeat_classifier_training(feature_frame, min_training_orders, repeat_horizon_days)
    if len(classifier_training) < 100:
        classifier_training = feature_frame[feature_frame["order_count_so_far"] >= min_training_orders].copy()
        classifier_training["repeat_purchase_target"] = classifier_training["days_until_next"].notna().astype("int64")

    repeat_model, days_model, quantity_model = build_models()

    classifier_train, classifier_valid, classifier_split_date = time_based_split(classifier_training)
    if classifier_train["repeat_purchase_target"].nunique() < 2:
        repeat_model = DummyClassifier(strategy="most_frequent")
    repeat_model.fit(classifier_train[FEATURE_COLUMNS], classifier_train["repeat_purchase_target"])

    reg_train, reg_valid, regression_split_date = time_based_split(regression_training)
    days_model.fit(reg_train[FEATURE_COLUMNS], reg_train["days_until_next"])
    quantity_model.fit(reg_train[FEATURE_COLUMNS], winsorized_quantity_target(reg_train))

    date_model_valid_days = round_to_week_days(np.clip(days_model.predict(reg_valid[FEATURE_COLUMNS]), 1, 365))
    date_baseline_valid_days = baseline_days_prediction(reg_valid)
    date_model_mae = float(mean_absolute_error(reg_valid["days_until_next"], date_model_valid_days))
    baseline_date_mae = float(mean_absolute_error(reg_valid["days_until_next"], date_baseline_valid_days))
    date_prediction_method = (
        "model_regression" if date_model_mae < baseline_date_mae else "median_interval_baseline"
    )
    validation = add_validation_predictions(reg_valid, repeat_model, days_model, quantity_model, date_prediction_method)

    classifier_valid = classifier_valid.copy()
    classifier_valid_probability = (
        positive_class_probability(repeat_model, classifier_valid) if not classifier_valid.empty else np.array([])
    )
    classifier_valid_pred = (classifier_valid_probability >= 0.5).astype("int64")
    classifier_target = classifier_valid["repeat_purchase_target"] if not classifier_valid.empty else pd.Series(dtype=int)
    classifier_auc = float("nan")
    classifier_average_precision = float("nan")
    classifier_accuracy = float("nan")
    classifier_precision = float("nan")
    classifier_recall = float("nan")
    classifier_f1 = float("nan")
    classifier_confusion_matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    positive_class_ratio = float("nan")
    if not classifier_valid.empty:
        positive_class_ratio = float(classifier_target.mean())
        classifier_accuracy = float(accuracy_score(classifier_target, classifier_valid_pred))
        classifier_precision = float(precision_score(classifier_target, classifier_valid_pred, zero_division=0))
        classifier_recall = float(recall_score(classifier_target, classifier_valid_pred, zero_division=0))
        classifier_f1 = float(f1_score(classifier_target, classifier_valid_pred, zero_division=0))
        if classifier_target.nunique() > 1:
            tn, fp, fn, tp = confusion_matrix(classifier_target, classifier_valid_pred, labels=[0, 1]).ravel()
            classifier_confusion_matrix = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        if classifier_target.nunique() > 1:
            classifier_auc = float(roc_auc_score(classifier_target, classifier_valid_probability))
            classifier_average_precision = float(
                average_precision_score(classifier_target, classifier_valid_probability)
            )

    baseline_quantity = validation["quantity_baseline"]
    baseline_quantity_mae = float(mean_absolute_error(validation["next_toplam_miktar"], baseline_quantity))
    selected_date_mae = float(mean_absolute_error(validation["days_until_next"], validation["date_predicted_days"]))
    quantity_model_mae = float(mean_absolute_error(validation["next_toplam_miktar"], validation["quantity_predicted"]))
    date_improvement_pct = (
        ((baseline_date_mae - selected_date_mae) / baseline_date_mae) * 100 if baseline_date_mae else float("nan")
    )
    raw_date_model_improvement_pct = (
        ((baseline_date_mae - date_model_mae) / baseline_date_mae) * 100 if baseline_date_mae else float("nan")
    )
    quantity_improvement_pct = (
        ((baseline_quantity_mae - quantity_model_mae) / baseline_quantity_mae) * 100
        if baseline_quantity_mae
        else float("nan")
    )

    metrics: dict[str, Any] = {
        "repeat_horizon_days": int(repeat_horizon_days),
        "data_max_hafta": feature_frame["hafta"].max().date().isoformat(),
        "classifier_training_rows": int(len(classifier_train)),
        "classifier_validation_rows": int(len(classifier_valid)),
        "classifier_split_date": classifier_split_date.date().isoformat(),
        "classifier_accuracy": classifier_accuracy,
        "classifier_precision": classifier_precision,
        "classifier_recall": classifier_recall,
        "classifier_f1": classifier_f1,
        "classifier_roc_auc": classifier_auc,
        "classifier_average_precision": classifier_average_precision,
        "classifier_positive_class_ratio": positive_class_ratio,
        "confusion_matrix_tn": classifier_confusion_matrix["tn"],
        "confusion_matrix_fp": classifier_confusion_matrix["fp"],
        "confusion_matrix_fn": classifier_confusion_matrix["fn"],
        "confusion_matrix_tp": classifier_confusion_matrix["tp"],
        "training_rows": int(len(reg_train)),
        "validation_rows": int(len(reg_valid)),
        "regression_split_date": regression_split_date.date().isoformat(),
        "date_prediction_method": date_prediction_method,
        "baseline_tarih_mae": baseline_date_mae,
        "raw_model_tarih_mae": date_model_mae,
        "baseline_miktar_mae": baseline_quantity_mae,
        "raw_tarih_model_iyilesme_yuzde": raw_date_model_improvement_pct,
        "kullanilan_tarih_iyilesme_yuzde": date_improvement_pct,
        "miktar_model_iyilesme_yuzde": quantity_improvement_pct,
        "tarih_mae": float(mean_absolute_error(validation["days_until_next"], validation["date_predicted_days"])),
        "tarih_rmse": float(
            np.sqrt(mean_squared_error(validation["days_until_next"], validation["date_predicted_days"]))
        ),
        "miktar_mae": float(mean_absolute_error(validation["next_toplam_miktar"], validation["quantity_predicted"])),
        "miktar_rmse": float(
            np.sqrt(mean_squared_error(validation["next_toplam_miktar"], validation["quantity_predicted"]))
        ),
        "tarih_r2": safe_r2(validation["days_until_next"], validation["date_predicted_days"]),
        "miktar_r2": safe_r2(validation["next_toplam_miktar"], validation["quantity_predicted"]),
        "tarih_smape": smape(validation["days_until_next"], validation["date_predicted_days"]),
        "miktar_smape": smape(validation["next_toplam_miktar"], validation["quantity_predicted"]),
    }

    analytics = {
        "hata_analizi": build_error_analysis(validation),
        "feature_importance": build_feature_importance(
            repeat_model,
            days_model,
            quantity_model,
            classifier_valid,
            reg_valid,
        ),
    }

    repeat_model.fit(classifier_training[FEATURE_COLUMNS], classifier_training["repeat_purchase_target"])
    days_model.fit(regression_training[FEATURE_COLUMNS], regression_training["days_until_next"])
    quantity_model.fit(regression_training[FEATURE_COLUMNS], winsorized_quantity_target(regression_training))
    return repeat_model, days_model, quantity_model, metrics, regression_training, analytics


def calculate_repeat_probability(latest: pd.DataFrame, repeat_model: Any) -> np.ndarray:
    model_probability = positive_class_probability(repeat_model, latest)
    order_count = latest["order_count_so_far"].to_numpy(dtype=float)
    regularity = (1 / (1 + latest["interval_cv"].to_numpy(dtype=float))).clip(0, 1)
    regularity = np.where(order_count >= 3, regularity, 0.35)
    avg_interval = latest["avg_interval_days"].replace(0, np.nan).fillna(PASSIVE_PRODUCT_DAYS).to_numpy(dtype=float)
    days_since_last = latest["son_siparisten_gecen_gun"].to_numpy(dtype=float)
    overdue_days = np.maximum(0, days_since_last - (avg_interval * 1.5))
    recency_factor = np.exp(-overdue_days / 120)
    history_factor = np.select([order_count >= 6, order_count >= 3], [1.0, 0.78], default=0.45)
    probability = model_probability * (0.65 + 0.35 * regularity) * recency_factor * history_factor
    return np.clip(probability, 0, 1)


def calculate_confidence_score(latest: pd.DataFrame, repeat_probability: np.ndarray) -> np.ndarray:
    order_count = latest["order_count_so_far"].to_numpy(dtype=float)
    history_score = np.minimum(order_count / 6, 1)
    regularity = (1 / (1 + latest["interval_cv"].to_numpy(dtype=float))).clip(0, 1)
    regularity = np.where(order_count >= 3, regularity, 0.25)
    activity_score = np.clip(1 - latest["son_siparisten_gecen_gun"].to_numpy(dtype=float) / 180, 0, 1)
    score = 100 * (
        0.35 * history_score
        + 0.25 * regularity
        + 0.20 * activity_score
        + 0.20 * np.asarray(repeat_probability, dtype=float)
    )
    caps = np.select([order_count <= 2, order_count <= 5], [45, 75], default=100)
    return np.minimum(score, caps).round(2)


def status_for_predictions(
    latest: pd.DataFrame,
    overdue_mask: np.ndarray,
    confidence_score: np.ndarray,
    repeat_probability: np.ndarray,
) -> pd.Series:
    order_count = latest["order_count_so_far"].to_numpy(dtype=int)
    days_since_last = latest["son_siparisten_gecen_gun"].to_numpy(dtype=int)
    status = np.full(len(latest), STATUS_VALID, dtype=object)
    status[((repeat_probability < ACTION_MIN_REPEAT_PROBABILITY) | (confidence_score < ACTION_MIN_CONFIDENCE_SCORE)) & (order_count >= 3)] = STATUS_LOW_PROBABILITY
    status[overdue_mask & (order_count >= 3) & (repeat_probability >= ACTION_MIN_REPEAT_PROBABILITY)] = STATUS_OVERDUE
    status[order_count <= 2] = STATUS_INSUFFICIENT
    status[days_since_last > PASSIVE_PRODUCT_DAYS] = STATUS_PASSIVE
    return pd.Series(status, index=latest.index, dtype="object")


def build_prediction_explanations(latest: pd.DataFrame) -> pd.Series:
    explanations = []
    for row in latest.itertuples(index=False):
        parts = [row.tahmin_aciklamasi_base]
        if row.ham_tahmin_gecmise_dustu:
            parts.append(
                "Ham model tarihi veri setinin son haftasından önce kaldığı için tarih ileriye taşındı."
            )
        if row.tahmin_durumu == STATUS_PASSIVE:
            parts.append(f"Son sipariş {int(row.son_siparisten_gecen_gun)} gün önce; pasif ürün olarak izlenmeli.")
        elif row.tahmin_durumu == STATUS_INSUFFICIENT:
            parts.append("1-2 sipariş olduğu için tahmin ana aksiyon listesinde düşük önceliklidir.")
        elif row.tahmin_durumu == STATUS_LOW_PROBABILITY:
            parts.append("Model güveni sınırlı; satış ekibi kontrolü önerilir.")
        elif row.tahmin_durumu == STATUS_OVERDUE:
            parts.append("Zamanlama modeli gecikmiş talep işareti verdi; ileri tarih kuralı uygulandı.")
        else:
            parts.append("Tahmin veri sonrasına düşüyor ve güven filtresinden geçti.")
        parts.append(f"Tekrar alma olasılığı %{row.tekrar_alma_olasiligi * 100:.1f}; güven skoru {row.guven_skoru:.1f}.")
        return_text = " ".join(parts)
        explanations.append(return_text)
    return pd.Series(explanations, index=latest.index, dtype="object")


def build_prediction_explanations(latest: pd.DataFrame) -> pd.Series:
    explanations = []
    for row in latest.itertuples(index=False):
        active_text = (
            f"Urun aktif: son siparis {int(row.son_siparisten_gecen_gun)} gun once."
            if row.son_siparisten_gecen_gun <= PASSIVE_PRODUCT_DAYS
            else f"Urun pasif: son siparis {int(row.son_siparisten_gecen_gun)} gun once."
        )
        confidence_reason = (
            f"Guven seviyesi {row.guven_seviyesi}; {int(row.gecmis_siparis_sayisi)} gecmis siparis, "
            f"duzenlilik skoru {float(row.siparis_duzenlilik_skoru):.2f}, guven skoru {float(row.guven_skoru):.1f}."
        )
        if pd.isna(row.tahmini_miktar):
            quantity_reason = "Tekrar alma olasiligi dusuk oldugu icin kesin miktar tahmini verilmedi."
        else:
            quantity_ratio = float(row.tahmini_miktar) / max(float(row.medyan_miktar), 1.0)
            if quantity_ratio <= 1.5:
                quantity_reason = (
                    f"Miktar tahmini gecmis medyana yakin: tahmin {row.tahmini_miktar}, "
                    f"medyan {row.medyan_miktar}, p95 ust sinir {row.miktar_p95}."
                )
            else:
                quantity_reason = (
                    f"Miktar tahmini medyanin uzerinde ama p95 ile sinirlandi: tahmin {row.tahmini_miktar}, "
                    f"medyan {row.medyan_miktar}, p95 {row.miktar_p95}."
                )

        date_reason = (
            "Tarih medyan siparis araligi baseline'i ile belirlendi."
            if row.tarih_tahmin_yontemi == "median_interval_baseline"
            else "Tarih regression modeli ile belirlendi."
        )
        if getattr(row, "tarih_yayma_uygulandi", False):
            date_reason += " Aksiyon listesinde ayni haftaya yigilmamasi icin medyan aralik penceresinde ileri haftaya dagitildi."
        pattern_reason = (
            f"Neden tahmin edildi: cari-urun ciftinde {int(row.gecmis_siparis_sayisi)} siparis var; "
            f"medyan aralik {row.medyan_siparis_araligi_gun} gun, ortalama aralik {row.ortalama_siparis_araligi_gun} gun."
        )
        parts = [pattern_reason, active_text, confidence_reason, date_reason, quantity_reason]
        if row.tahmin_durumu == STATUS_PASSIVE:
            parts.append("Ana aksiyon listesine alinmadi; pasif urun sayfasinda takip edilmeli.")
        elif row.tahmin_durumu == STATUS_INSUFFICIENT:
            parts.append("1-2 siparis oldugu icin kesin aksiyon tahmini uretilmedi.")
        elif row.tahmin_durumu == STATUS_LOW_PROBABILITY:
            parts.append(
                f"Tekrar alma olasiligi %{row.tekrar_alma_olasiligi * 100:.1f}; esik %{ACTION_MIN_REPEAT_PROBABILITY * 100:.0f} altinda veya guven skoru dusuk."
            )
        elif row.tahmin_durumu == STATUS_OVERDUE:
            parts.append("Beklenen siparis tarihi veri son haftasini gecmis; gelecege itilmedi, overdue olarak ayrildi.")
        else:
            parts.append("Aksiyon filtresinden gecti: aktif, yeterli gecmise sahip, olasilik ve guven skoru esik uzerinde.")
        parts.append(f"Tekrar alma olasiligi %{row.tekrar_alma_olasiligi * 100:.1f}.")
        explanations.append(" ".join(parts))
    return pd.Series(explanations, index=latest.index, dtype="object")


def create_predictions(
    feature_frame: pd.DataFrame,
    repeat_model: Any,
    days_model: HistGradientBoostingRegressor,
    quantity_model: TransformedTargetRegressor,
    metrics: dict[str, Any],
) -> pd.DataFrame:
    latest_indexes = feature_frame.groupby(["cari_id", "stock_id"], sort=False)["hafta"].idxmax()
    latest = feature_frame.loc[latest_indexes].copy()
    data_max_date = pd.Timestamp(feature_frame["hafta"].max())

    model_days = round_to_week_days(np.clip(days_model.predict(latest[FEATURE_COLUMNS]), 1, 365))
    baseline_days = baseline_days_prediction(latest)
    date_prediction_method = metrics.get("date_prediction_method", "model_regression")
    selected_days = baseline_days if date_prediction_method == "median_interval_baseline" else model_days
    selected_date = latest["hafta"] + pd.to_timedelta(selected_days, unit="D")
    model_date = latest["hafta"] + pd.to_timedelta(model_days, unit="D")
    baseline_date = latest["hafta"] + pd.to_timedelta(baseline_days, unit="D")
    overdue_mask = (selected_date <= data_max_date).to_numpy()
    predicted_quantity = predict_quantities(latest, quantity_model)

    pattern_details = latest.apply(describe_pair_pattern, axis=1, result_type="expand")
    pattern_details.columns = ["siparis_deseni", "aktif_aylar", "tahmin_aciklamasi_base"]

    latest["son_siparis_tarihi"] = latest["hafta"]
    latest["son_siparis_miktari"] = latest["toplam_miktar"].round(2)
    latest["gecmis_siparis_sayisi"] = latest["order_count_so_far"].astype("int64")
    latest["son_siparisten_gecen_gun"] = (data_max_date - latest["hafta"]).dt.days.astype("int64")
    latest["tahmini_gun"] = selected_days.astype("int64")
    latest["tahmini_miktar"] = predicted_quantity
    latest["tahmini_siparis_tarihi"] = selected_date
    latest["model_tahmini_siparis_tarihi"] = model_date
    latest["baseline_tahmini_siparis_tarihi"] = baseline_date
    latest["ham_tahmini_siparis_tarihi"] = model_date
    latest["ham_tahmin_gecmise_dustu"] = overdue_mask
    latest["tarih_tahmin_yontemi"] = date_prediction_method

    avg_interval, regularity_score, interval_note = describe_interval_reliability(latest)
    latest["ortalama_siparis_araligi_gun"] = avg_interval
    latest["medyan_siparis_araligi_gun"] = latest["median_interval_days"].round(1)
    latest["ortalama_miktar"] = latest["qty_mean_so_far"].round(2)
    latest["medyan_miktar"] = latest["qty_median_so_far"].round(2)
    latest["son3_siparis_ortalamasi"] = latest["qty_last3_mean_so_far"].round(2)
    latest["son6_siparis_ortalamasi"] = latest["qty_last6_mean_so_far"].round(2)
    latest["siparis_araligi_std"] = latest["interval_std_days"].round(2)
    latest["miktar_std"] = latest["qty_std_so_far"].round(2)
    latest["miktar_p95"] = latest["qty_p95_so_far"].round(2)
    latest["siparis_duzenlilik_skoru"] = regularity_score.fillna(0).round(3)
    latest["aralik_yorumu"] = interval_note
    latest[["siparis_deseni", "aktif_aylar", "tahmin_aciklamasi_base"]] = pattern_details
    latest["guven_seviyesi"] = confidence_level_from_count(latest["gecmis_siparis_sayisi"].reset_index(drop=True)).values
    latest["tekrar_alma_olasiligi"] = calculate_repeat_probability(latest, repeat_model).round(4)
    latest["guven_skoru"] = calculate_confidence_score(latest, latest["tekrar_alma_olasiligi"])
    latest["tahmin_durumu"] = status_for_predictions(
        latest,
        overdue_mask,
        latest["guven_skoru"].to_numpy(),
        latest["tekrar_alma_olasiligi"].to_numpy(),
    )
    spread_mask = (
        (latest["tahmin_durumu"] == STATUS_VALID)
        & (latest["tekrar_alma_olasiligi"] >= ACTION_MIN_REPEAT_PROBABILITY)
        & (latest["guven_skoru"] >= ACTION_MIN_CONFIDENCE_SCORE)
        & (latest["gecmis_siparis_sayisi"] >= 3)
        & (latest["son_siparisten_gecen_gun"] <= PASSIVE_PRODUCT_DAYS)
        & (latest["tahmini_siparis_tarihi"] > data_max_date)
    )
    latest["tarih_yayma_uygulandi"] = False
    if spread_mask.any():
        spread_indexes = latest.index[spread_mask]
        interval_weeks = np.ceil(
            latest.loc[spread_mask, "medyan_siparis_araligi_gun"].fillna(7).clip(lower=7) / 7
        ).astype("int64")
        interval_weeks = interval_weeks.clip(lower=1, upper=12)
        pair_hash = (
            latest.loc[spread_mask, "cari_id"].astype("int64") * 1_000_003
            + latest.loc[spread_mask, "stock_id"].astype("int64")
        ).abs()
        spread_offsets = pair_hash.to_numpy(dtype="int64") % interval_weeks.to_numpy(dtype="int64")
        current_future_weeks = np.ceil(
            (
                pd.to_datetime(latest.loc[spread_mask, "tahmini_siparis_tarihi"]) - data_max_date
            ).dt.days.clip(lower=7)
            / 7
        ).astype("int64")
        spread_future_weeks = current_future_weeks.to_numpy(dtype="int64") + spread_offsets
        spread_dates = data_max_date + pd.to_timedelta(spread_future_weeks * 7, unit="D")
        latest.loc[spread_mask, "tahmini_siparis_tarihi"] = spread_dates.to_numpy()
        latest.loc[spread_mask, "tahmini_gun"] = (
            latest.loc[spread_mask, "tahmini_siparis_tarihi"] - latest.loc[spread_mask, "son_siparis_tarihi"]
        ).dt.days
        latest.loc[spread_indexes[spread_offsets > 0], "tarih_yayma_uygulandi"] = True
    low_probability_mask = latest["tahmin_durumu"] == STATUS_LOW_PROBABILITY
    latest.loc[low_probability_mask, ["tahmini_siparis_tarihi", "tahmini_gun", "tahmini_miktar"]] = pd.NA
    latest["aksiyon_tahmini"] = (
        (latest["tahmin_durumu"] == STATUS_VALID)
        & (latest["tekrar_alma_olasiligi"] >= ACTION_MIN_REPEAT_PROBABILITY)
        & (latest["guven_skoru"] >= ACTION_MIN_CONFIDENCE_SCORE)
        & (latest["gecmis_siparis_sayisi"] >= 3)
        & (latest["son_siparisten_gecen_gun"] <= PASSIVE_PRODUCT_DAYS)
        & (latest["tahmini_siparis_tarihi"] > data_max_date)
    )
    latest["tahmin_aciklamasi"] = build_prediction_explanations(latest)

    output_columns = [
        "cari_id",
        "cari_kod",
        "stock_id",
        "stock_kod",
        "stock_ad",
        "son_siparis_tarihi",
        "son_siparis_miktari",
        "gecmis_siparis_sayisi",
        "son_siparisten_gecen_gun",
        "tahmini_siparis_tarihi",
        "tahmini_gun",
        "tahmini_miktar",
        "tekrar_alma_olasiligi",
        "guven_skoru",
        "guven_seviyesi",
        "tahmin_durumu",
        "tahmin_aciklamasi",
        "ortalama_siparis_araligi_gun",
        "medyan_siparis_araligi_gun",
        "ortalama_miktar",
        "medyan_miktar",
        "son3_siparis_ortalamasi",
        "son6_siparis_ortalamasi",
        "siparis_araligi_std",
        "miktar_std",
        "miktar_p95",
        "aktif_aylar",
        "siparis_duzenlilik_skoru",
        "aralik_yorumu",
        "siparis_deseni",
        "tarih_tahmin_yontemi",
        "model_tahmini_siparis_tarihi",
        "baseline_tahmini_siparis_tarihi",
        "tarih_yayma_uygulandi",
        "ham_tahmini_siparis_tarihi",
        "ham_tahmin_gecmise_dustu",
        "aksiyon_tahmini",
    ]
    predictions = latest[output_columns].copy()
    predictions["tahmin_durumu_sira"] = predictions["tahmin_durumu"].map(STATUS_PRIORITY).fillna(99)
    predictions = predictions.sort_values(
        ["tahmin_durumu_sira", "guven_skoru", "tahmini_siparis_tarihi", "cari_kod", "stock_kod", "stock_ad"],
        ascending=[True, False, True, True, True, True],
    ).drop(columns="tahmin_durumu_sira")
    metrics["prediction_rows"] = int(len(predictions))
    metrics["overdue_rows"] = int((predictions["tahmin_durumu"] == STATUS_OVERDUE).sum())
    metrics["passive_product_rows"] = int((predictions["tahmin_durumu"] == STATUS_PASSIVE).sum())
    metrics["actionable_prediction_rows"] = int(predictions["aksiyon_tahmini"].sum())
    metrics["low_probability_rows"] = int((predictions["tahmin_durumu"] == STATUS_LOW_PROBABILITY).sum())
    return predictions.reset_index(drop=True)


def write_predictions_to_postgres(predictions: pd.DataFrame, table: str, schema: str | None) -> None:
    engine = make_postgres_engine()
    prepare_table(engine, table, schema, "predictions", "replace")
    predictions.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=20_000,
        method="multi",
    )
    print(f"Predictions written to {table_reference(table, schema)}.")


def main() -> None:
    args = parse_args()
    if args.source == "postgres":
        sales = read_sales_from_postgres(args.table, args.schema)
    else:
        sales = load_sales_excel(args.excel)

    feature_frame = add_time_series_features(sales)
    repeat_model, days_model, quantity_model, metrics, training, analytics = train_models(
        feature_frame,
        args.min_training_orders,
        args.repeat_horizon_days,
    )
    predictions = create_predictions(feature_frame, repeat_model, days_model, quantity_model, metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_predictions_to_excel(predictions, output_path, metrics=metrics, analytics=analytics)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "repeat_model": repeat_model,
            "days_model": days_model,
            "quantity_model": quantity_model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "analytics": analytics,
            "trained_rows": len(training),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )

    if args.write_postgres:
        write_predictions_to_postgres(predictions, args.predictions_table, args.schema)

    print(f"Training rows: {metrics['training_rows']:,}")
    print(f"Validation rows: {metrics['validation_rows']:,}")
    print(f"Validation next-order date MAE: {metrics['tarih_mae']:.2f} days")
    print(f"Validation quantity MAE: {metrics['miktar_mae']:.2f}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Actionable prediction rows: {metrics['actionable_prediction_rows']:,}")
    print(f"Overdue rows: {metrics['overdue_rows']:,}")
    print(f"Excel written: {output_path}")
    print(f"Model written: {model_path}")


if __name__ == "__main__":
    main()
