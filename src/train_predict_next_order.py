from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.common import (
    DEFAULT_EXCEL_PATH,
    aggregate_weekly_sales,
    clean_sales_frame,
    load_sales_excel,
    make_postgres_engine,
    table_reference,
)
from src.db_schema import prepare_table


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
    "order_count_so_far",
    "days_since_first_order",
    "last_interval_days",
    "avg_interval_days",
    "qty_mean_so_far",
    "amount_mean_so_far",
    "line_mean_so_far",
    "qty_sum_so_far",
    "amount_sum_so_far",
]


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
    return parser.parse_args()


def read_sales_from_postgres(table: str, schema: str | None) -> pd.DataFrame:
    engine = make_postgres_engine()
    query = f"SELECT * FROM {table_reference(table, schema)}"
    frame = pd.read_sql_query(query, engine)
    return clean_sales_frame(frame)


def add_time_series_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = aggregate_weekly_sales(frame)
    df = df.sort_values(["cari_id", "stock_id", "hafta"]).reset_index(drop=True)
    groups = df.groupby(["cari_id", "stock_id"], sort=False)

    df["next_hafta"] = groups["hafta"].shift(-1)
    df["days_until_next"] = (df["next_hafta"] - df["hafta"]).dt.days
    df["next_toplam_miktar"] = groups["toplam_miktar"].shift(-1)

    df["first_hafta"] = groups["hafta"].transform("first")
    df["order_count_so_far"] = groups.cumcount() + 1
    df["days_since_first_order"] = (df["hafta"] - df["first_hafta"]).dt.days
    df["last_interval_days"] = groups["hafta"].diff().dt.days
    df["interval_count_so_far"] = (df["order_count_so_far"] - 1).clip(lower=0)
    df["interval_sum_so_far"] = groups["last_interval_days"].cumsum()
    df["avg_interval_days"] = df["interval_sum_so_far"] / df["interval_count_so_far"].replace(0, np.nan)

    df["qty_sum_so_far"] = groups["toplam_miktar"].cumsum()
    df["amount_sum_so_far"] = groups["toplam_tutar"].cumsum()
    df["line_sum_so_far"] = groups["siparis_satir_sayisi"].cumsum()
    df["qty_mean_so_far"] = df["qty_sum_so_far"] / df["order_count_so_far"]
    df["amount_mean_so_far"] = df["amount_sum_so_far"] / df["order_count_so_far"]
    df["line_mean_so_far"] = df["line_sum_so_far"] / df["order_count_so_far"]
    df["unit_price"] = np.where(df["toplam_miktar"] > 0, df["toplam_tutar"] / df["toplam_miktar"], 0)

    iso_calendar = df["hafta"].dt.isocalendar()
    df["weekofyear"] = iso_calendar.week.astype("int64")
    df["month"] = df["hafta"].dt.month
    df["quarter"] = df["hafta"].dt.quarter
    df["year"] = df["hafta"].dt.year

    global_interval_median = df["last_interval_days"].median()
    if pd.isna(global_interval_median):
        global_interval_median = 28
    df["last_interval_days"] = df["last_interval_days"].fillna(global_interval_median)
    df["avg_interval_days"] = df["avg_interval_days"].fillna(df["last_interval_days"])
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def build_models() -> tuple[HistGradientBoostingRegressor, TransformedTargetRegressor]:
    days_model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        l2_regularization=0.05,
        min_samples_leaf=25,
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
    return days_model, quantity_model


def train_models(feature_frame: pd.DataFrame, min_training_orders: int):
    training = feature_frame[
        (feature_frame["days_until_next"].notna())
        & (feature_frame["days_until_next"] > 0)
        & (feature_frame["order_count_so_far"] >= min_training_orders)
    ].copy()
    if len(training) < 100:
        raise ValueError("Not enough historical repeated orders to train a reliable model.")

    training = training.sort_values("hafta").reset_index(drop=True)
    split_index = int(len(training) * 0.8)
    split_index = min(max(split_index, 1), len(training) - 1)

    train = training.iloc[:split_index]
    valid = training.iloc[split_index:]
    x_train = train[FEATURE_COLUMNS]
    x_valid = valid[FEATURE_COLUMNS]

    days_model, quantity_model = build_models()
    days_model.fit(x_train, train["days_until_next"])
    quantity_model.fit(x_train, train["next_toplam_miktar"].clip(lower=0))

    valid_days_pred = np.clip(days_model.predict(x_valid), 1, 365)
    valid_qty_pred = np.clip(quantity_model.predict(x_valid), 0, None)
    metrics = {
        "training_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "validation_days_mae": float(mean_absolute_error(valid["days_until_next"], valid_days_pred)),
        "validation_quantity_mae": float(mean_absolute_error(valid["next_toplam_miktar"], valid_qty_pred)),
    }

    days_model.fit(training[FEATURE_COLUMNS], training["days_until_next"])
    quantity_model.fit(training[FEATURE_COLUMNS], training["next_toplam_miktar"].clip(lower=0))
    return days_model, quantity_model, metrics, training


def create_predictions(feature_frame: pd.DataFrame, days_model, quantity_model, metrics: dict) -> pd.DataFrame:
    latest_indexes = feature_frame.groupby(["cari_id", "stock_id"], sort=False)["hafta"].idxmax()
    latest = feature_frame.loc[latest_indexes].copy()

    predicted_days = np.clip(days_model.predict(latest[FEATURE_COLUMNS]), 1, 365)
    predicted_days = np.maximum(7, np.round(predicted_days / 7) * 7).astype(int)
    predicted_quantity = np.clip(quantity_model.predict(latest[FEATURE_COLUMNS]), 0, None)
    predicted_quantity = np.maximum(1, np.floor(predicted_quantity + 0.5)).astype("int64")

    latest["tahmini_gun"] = predicted_days
    latest["tahmini_miktar"] = predicted_quantity
    latest["tahmini_siparis_tarihi"] = latest["hafta"] + pd.to_timedelta(latest["tahmini_gun"], unit="D")

    output_columns = [
        "cari_id",
        "cari_kod",
        "stock_id",
        "stock_kod",
        "stock_ad",
        "tahmini_siparis_tarihi",
        "tahmini_miktar",
    ]
    predictions = latest[output_columns].sort_values(["cari_kod", "stock_kod", "stock_ad"])
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
    days_model, quantity_model, metrics, training = train_models(feature_frame, args.min_training_orders)
    predictions = create_predictions(feature_frame, days_model, quantity_model, metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        excel_predictions = predictions.rename(
            columns={
                "cari_id": "CariId",
                "cari_kod": "CariKod",
                "stock_id": "StockId",
                "stock_kod": "StockKod",
                "stock_ad": "StockAd",
                "tahmini_siparis_tarihi": "TahminiSiparisTarihi",
                "tahmini_miktar": "TahminiMiktar",
            }
        )
        excel_predictions["TahminiSiparisTarihi"] = pd.to_datetime(
            excel_predictions["TahminiSiparisTarihi"]
        ).dt.strftime("%Y-%m-%d")
        excel_predictions.to_excel(writer, sheet_name="tahminler", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="model_metrics", index=False)

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "days_model": days_model,
            "quantity_model": quantity_model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "trained_rows": len(training),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )

    if args.write_postgres:
        write_predictions_to_postgres(predictions, args.predictions_table, args.schema)

    print(f"Training rows: {metrics['training_rows']:,}")
    print(f"Validation rows: {metrics['validation_rows']:,}")
    print(f"Validation next-order date MAE: {metrics['validation_days_mae']:.2f} days")
    print(f"Validation quantity MAE: {metrics['validation_quantity_mae']:.2f}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Excel written: {output_path}")
    print(f"Model written: {model_path}")


if __name__ == "__main__":
    main()
