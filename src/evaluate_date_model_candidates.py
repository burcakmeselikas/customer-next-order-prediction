from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.train_predict_next_order import FEATURE_COLUMNS, add_time_series_features, read_sales_from_postgres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare next-order date model candidates.")
    parser.add_argument("--table", default="historical_sales_weekly")
    parser.add_argument("--schema", default=None)
    parser.add_argument("--min-training-orders", type=int, default=2)
    return parser.parse_args()


def rounded_week_days(values) -> np.ndarray:
    return np.maximum(7, np.round(np.asarray(values, dtype=float) / 7) * 7).astype("int64")


def metric_row(name: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "model": name,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def main() -> None:
    args = parse_args()
    sales = read_sales_from_postgres(args.table, args.schema)
    feature_frame = add_time_series_features(sales)
    training = feature_frame[
        (feature_frame["days_until_next"].notna())
        & (feature_frame["days_until_next"] > 0)
        & (feature_frame["order_count_so_far"] >= args.min_training_orders)
    ].copy()
    training = training.sort_values("hafta").reset_index(drop=True)
    split_index = int(len(training) * 0.8)
    split_index = min(max(split_index, 1), len(training) - 1)
    train = training.iloc[:split_index]
    valid = training.iloc[split_index:]
    x_train = train[FEATURE_COLUMNS]
    x_valid = valid[FEATURE_COLUMNS]
    actual_days = valid["days_until_next"].to_numpy(dtype=float)

    candidates = []

    absolute_regressor = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=320,
        learning_rate=0.045,
        l2_regularization=0.05,
        max_leaf_nodes=47,
        min_samples_leaf=18,
        random_state=42,
    )
    absolute_regressor.fit(x_train, train["days_until_next"])
    absolute_pred = np.clip(absolute_regressor.predict(x_valid), 1, 365)
    candidates.append(metric_row("absolute_regressor_raw", actual_days, absolute_pred))
    candidates.append(metric_row("absolute_regressor_week_rounded", actual_days, rounded_week_days(absolute_pred)))

    classifier = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.055,
        l2_regularization=0.05,
        max_leaf_nodes=47,
        min_samples_leaf=18,
        random_state=45,
    )
    train_weeks = rounded_week_days(train["days_until_next"]) // 7
    classifier.fit(x_train, train_weeks)
    classifier_proba = classifier.predict_proba(x_valid)
    classifier_confidence = classifier_proba.max(axis=1)
    classifier_pred = classifier.classes_[classifier_proba.argmax(axis=1)] * 7
    candidates.append(metric_row("week_classifier", actual_days, classifier_pred))
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.60, 0.70]:
        hybrid_pred = np.where(classifier_confidence >= threshold, classifier_pred, absolute_pred)
        candidates.append(metric_row(f"hybrid_classifier_conf_{threshold:.2f}", actual_days, hybrid_pred))

    for row in sorted(candidates, key=lambda item: item["mae"]):
        print(f"{row['model']}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
