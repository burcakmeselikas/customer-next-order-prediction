from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "TblGecmisSatisVerileri_haftalik_3.xlsx"

RAW_TO_SNAKE_COLUMNS = {
    "CariId": "cari_id",
    "CariKod": "cari_kod",
    "StockId": "stock_id",
    "StockKod": "stock_kod",
    "StockAd": "stock_ad",
    "Hafta": "hafta",
    "ToplamMiktar": "toplam_miktar",
    "ToplamTutar": "toplam_tutar",
    "SiparisSatirSayisi": "siparis_satir_sayisi",
}

REQUIRED_COLUMNS = list(RAW_TO_SNAKE_COLUMNS.values())


def load_env_file(path: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing variables."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_sales_excel(path: str | Path = DEFAULT_EXCEL_PATH, nrows: int | None = None) -> pd.DataFrame:
    """Read the weekly sales Excel file and return a normalized DataFrame."""
    frame = pd.read_excel(path, engine="openpyxl", nrows=nrows)
    return clean_sales_frame(frame)


def clean_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, data types, and impossible numeric values."""
    df = frame.copy()
    df = df.rename(columns=RAW_TO_SNAKE_COLUMNS)
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["hafta"] = pd.to_datetime(df["hafta"], errors="coerce")

    numeric_columns = ["cari_id", "stock_id", "toplam_miktar", "toplam_tutar", "siparis_satir_sayisi"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = ["cari_kod", "stock_kod", "stock_ad"]
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()

    df = df.dropna(subset=["cari_id", "stock_id", "hafta"])
    df["cari_id"] = df["cari_id"].astype("int64")
    df["stock_id"] = df["stock_id"].astype("int64")
    df["toplam_miktar"] = df["toplam_miktar"].fillna(0).clip(lower=0)
    df["toplam_tutar"] = df["toplam_tutar"].fillna(0).clip(lower=0)
    df["siparis_satir_sayisi"] = df["siparis_satir_sayisi"].fillna(0).clip(lower=0).astype("int64")
    return df


def aggregate_weekly_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per customer-product-week."""
    df = clean_sales_frame(frame)
    group_columns = ["cari_id", "cari_kod", "stock_id", "stock_kod", "stock_ad", "hafta"]
    aggregated = (
        df.groupby(group_columns, as_index=False, observed=True)
        .agg(
            toplam_miktar=("toplam_miktar", "sum"),
            toplam_tutar=("toplam_tutar", "sum"),
            siparis_satir_sayisi=("siparis_satir_sayisi", "sum"),
        )
        .sort_values(["cari_id", "stock_id", "hafta"])
        .reset_index(drop=True)
    )
    return aggregated


def get_database_url() -> str | URL:
    """Build a SQLAlchemy PostgreSQL URL from environment variables."""
    load_env_file()
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url

    host = os.getenv("POSTGRES_HOST", "localhost").strip()
    port = os.getenv("POSTGRES_PORT", "5432").strip()
    database = os.getenv("POSTGRES_DB", "customer_next_order").strip()
    user = os.getenv("POSTGRES_USER", "postgres").strip()
    password = os.getenv("POSTGRES_PASSWORD", "").strip()

    if not all([host, port, database, user]):
        raise ValueError("Set DATABASE_URL or POSTGRES_HOST/PORT/DB/USER/PASSWORD environment variables.")

    return URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )


def make_postgres_engine():
    return create_engine(get_database_url())


def quote_identifier(identifier: str) -> str:
    """Quote a simple SQL identifier after validating it."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def table_reference(table: str, schema: str | None = None) -> str:
    table_sql = quote_identifier(table)
    if schema:
        return f"{quote_identifier(schema)}.{table_sql}"
    return table_sql
