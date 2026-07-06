from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common import make_postgres_engine, table_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PostgreSQL prediction table to Excel.")
    parser.add_argument("--table", default="next_order_predictions")
    parser.add_argument("--schema", default=None)
    parser.add_argument("--output", default="outputs/next_order_predictions_from_postgres.xlsx")
    return parser.parse_args()


def read_predictions(table: str, schema: str | None) -> pd.DataFrame:
    engine = make_postgres_engine()
    query = (
        f"SELECT * FROM {table_reference(table, schema)} "
        "ORDER BY cari_kod, stock_kod, stock_ad"
    )
    return pd.read_sql_query(query, engine)


def export_predictions_to_excel(predictions: pd.DataFrame, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        excel_predictions.to_excel(writer, sheet_name="tahminler", index=False)

    return output_path


def main() -> None:
    args = parse_args()
    predictions = read_predictions(args.table, args.schema)
    output_path = export_predictions_to_excel(predictions, args.output)
    print(f"Exported {len(predictions):,} prediction rows from {table_reference(args.table, args.schema)}.")
    print(f"Excel written: {output_path}")


if __name__ == "__main__":
    main()
