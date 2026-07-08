from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib

from src.common import DEFAULT_EXCEL_PATH, load_sales_excel, make_postgres_engine, table_reference
from src.db_schema import prepare_table
from src.export_predictions import export_predictions_to_excel, read_predictions
from src.train_predict_next_order import (
    FEATURE_COLUMNS,
    add_time_series_features,
    create_predictions,
    read_sales_from_postgres,
    train_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PostgreSQL tables, write predictions to PostgreSQL, then export Excel from PostgreSQL."
    )
    parser.add_argument("--excel", default=str(DEFAULT_EXCEL_PATH))
    parser.add_argument("--schema", default=None)
    parser.add_argument("--sales-table", default="historical_sales_weekly")
    parser.add_argument("--predictions-table", default="next_order_predictions")
    parser.add_argument("--output", default="outputs/next_order_predictions_from_postgres.xlsx")
    parser.add_argument("--model-output", default="models/next_order_models.joblib")
    parser.add_argument("--if-exists", default="replace", choices=["fail", "replace", "append"])
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--min-training-orders", type=int, default=2)
    parser.add_argument("--skip-sales-load", action="store_true", help="Use existing PostgreSQL sales table.")
    return parser.parse_args()


def write_frame_to_table(frame, engine, table: str, schema: str | None, chunksize: int) -> None:
    frame.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )


def main() -> None:
    args = parse_args()
    engine = make_postgres_engine()

    if not args.skip_sales_load:
        sales = load_sales_excel(args.excel)
        sales["loaded_at"] = datetime.now().replace(microsecond=0)
        prepare_table(engine, args.sales_table, args.schema, "historical", args.if_exists)
        write_frame_to_table(sales, engine, args.sales_table, args.schema, args.chunksize)
        print(f"Loaded {len(sales):,} rows into {table_reference(args.sales_table, args.schema)}.")

    sales_from_db = read_sales_from_postgres(args.sales_table, args.schema)
    feature_frame = add_time_series_features(sales_from_db)
    repeat_model, days_model, quantity_model, metrics, training, analytics = train_models(
        feature_frame,
        args.min_training_orders,
    )
    predictions = create_predictions(feature_frame, repeat_model, days_model, quantity_model, metrics)

    prepare_table(engine, args.predictions_table, args.schema, "predictions", "replace")
    write_frame_to_table(predictions, engine, args.predictions_table, args.schema, args.chunksize)
    print(f"Wrote {len(predictions):,} rows into {table_reference(args.predictions_table, args.schema)}.")

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
            "source_table": table_reference(args.sales_table, args.schema),
            "predictions_table": table_reference(args.predictions_table, args.schema),
            "created_at": datetime.now().replace(microsecond=0).isoformat(),
        },
        model_path,
    )

    predictions_from_db = read_predictions(args.predictions_table, args.schema)
    output_path = export_predictions_to_excel(predictions_from_db, args.output, metrics=metrics, analytics=analytics)

    print(f"Training rows: {metrics['training_rows']:,}")
    print(f"Validation rows: {metrics['validation_rows']:,}")
    print(f"Validation next-order date MAE: {metrics['tarih_mae']:.2f} days")
    print(f"Validation quantity MAE: {metrics['miktar_mae']:.2f}")
    print(f"Actionable prediction rows: {metrics['actionable_prediction_rows']:,}")
    print(f"Overdue rows: {metrics['overdue_rows']:,}")
    print(f"Excel exported from PostgreSQL: {output_path}")
    print(f"Model written: {model_path}")


if __name__ == "__main__":
    main()
