from __future__ import annotations

import argparse

import pandas as pd

from src.common import DEFAULT_EXCEL_PATH, load_sales_excel, make_postgres_engine, table_reference
from src.db_schema import prepare_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load weekly sales Excel data into PostgreSQL.")
    parser.add_argument("--excel", default=str(DEFAULT_EXCEL_PATH), help="Path to the source Excel file.")
    parser.add_argument("--table", default="historical_sales_weekly", help="Destination PostgreSQL table name.")
    parser.add_argument("--schema", default=None, help="Optional destination schema.")
    parser.add_argument("--if-exists", default="replace", choices=["fail", "replace", "append"])
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--sample-rows", type=int, default=None, help="Load only the first N rows for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Read and validate data without writing to PostgreSQL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sales = load_sales_excel(args.excel, nrows=args.sample_rows)
    sales["loaded_at"] = pd.Timestamp.now().replace(microsecond=0)

    print(f"Rows ready: {len(sales):,}")
    print(f"Date range: {sales['hafta'].min().date()} - {sales['hafta'].max().date()}")
    print(f"Customer count: {sales['cari_id'].nunique():,}")
    print(f"Product count: {sales['stock_id'].nunique():,}")

    if args.dry_run:
        print("Dry run complete. PostgreSQL was not changed.")
        return

    engine = make_postgres_engine()
    prepare_table(engine, args.table, args.schema, "historical", args.if_exists)
    sales.to_sql(
        name=args.table,
        con=engine,
        schema=args.schema,
        if_exists="append",
        index=False,
        chunksize=args.chunksize,
        method="multi",
    )
    print(f"Loaded {len(sales):,} rows into {table_reference(args.table, args.schema)}.")


if __name__ == "__main__":
    main()
