from __future__ import annotations

from sqlalchemy import text

from src.common import quote_identifier, table_reference


def create_schema_if_needed(engine, schema: str | None) -> None:
    if not schema:
        return
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema)}"))


def recreate_table(engine, table: str, schema: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"DROP TABLE IF EXISTS {table_reference(table, schema)}"))


def truncate_table(engine, table: str, schema: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE TABLE {table_reference(table, schema)}"))


def create_historical_sales_table(engine, table: str, schema: str | None = None) -> None:
    create_schema_if_needed(engine, schema)
    table_sql = table_reference(table, schema)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {table_sql} (
                    cari_id BIGINT NOT NULL,
                    cari_kod TEXT NOT NULL,
                    stock_id BIGINT NOT NULL,
                    stock_kod TEXT NOT NULL,
                    stock_ad TEXT NOT NULL,
                    hafta TIMESTAMP NOT NULL,
                    toplam_miktar NUMERIC(18, 4) NOT NULL,
                    toplam_tutar NUMERIC(18, 4) NOT NULL,
                    siparis_satir_sayisi BIGINT NOT NULL,
                    loaded_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (cari_id, stock_id, hafta)
                )
                """
            )
        )
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_hafta ON {table_sql} (hafta)"))
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_cari_id ON {table_sql} (cari_id)"))
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_stock_id ON {table_sql} (stock_id)"))


def create_next_order_predictions_table(engine, table: str, schema: str | None = None) -> None:
    create_schema_if_needed(engine, schema)
    table_sql = table_reference(table, schema)
    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {table_sql} (
                    cari_id BIGINT NOT NULL,
                    cari_kod TEXT NOT NULL,
                    stock_id BIGINT NOT NULL,
                    stock_kod TEXT NOT NULL,
                    stock_ad TEXT NOT NULL,
                    son_siparis_tarihi TIMESTAMP NOT NULL,
                    son_siparis_miktari NUMERIC(18, 2) NOT NULL,
                    gecmis_siparis_sayisi BIGINT NOT NULL,
                    son_siparisten_gecen_gun BIGINT NOT NULL,
                    tahmini_siparis_tarihi TIMESTAMP,
                    tahmini_gun BIGINT,
                    tahmini_miktar NUMERIC(18, 2),
                    tekrar_alma_olasiligi NUMERIC(18, 4) NOT NULL,
                    guven_skoru NUMERIC(18, 2) NOT NULL,
                    guven_seviyesi TEXT NOT NULL,
                    tahmin_durumu TEXT NOT NULL,
                    tahmin_aciklamasi TEXT NOT NULL,
                    ortalama_siparis_araligi_gun NUMERIC(18, 2),
                    medyan_siparis_araligi_gun NUMERIC(18, 2),
                    ortalama_miktar NUMERIC(18, 2) NOT NULL,
                    medyan_miktar NUMERIC(18, 2) NOT NULL,
                    son3_siparis_ortalamasi NUMERIC(18, 2) NOT NULL,
                    son6_siparis_ortalamasi NUMERIC(18, 2) NOT NULL,
                    siparis_araligi_std NUMERIC(18, 2) NOT NULL,
                    miktar_std NUMERIC(18, 2) NOT NULL,
                    miktar_p95 NUMERIC(18, 2) NOT NULL,
                    aktif_aylar TEXT NOT NULL,
                    siparis_duzenlilik_skoru NUMERIC(18, 3) NOT NULL,
                    aralik_yorumu TEXT NOT NULL,
                    siparis_deseni TEXT NOT NULL,
                    tarih_tahmin_yontemi TEXT NOT NULL,
                    model_tahmini_siparis_tarihi TIMESTAMP,
                    baseline_tahmini_siparis_tarihi TIMESTAMP,
                    tarih_yayma_uygulandi BOOLEAN NOT NULL,
                    ham_tahmini_siparis_tarihi TIMESTAMP NOT NULL,
                    ham_tahmin_gecmise_dustu BOOLEAN NOT NULL,
                    aksiyon_tahmini BOOLEAN NOT NULL,
                    PRIMARY KEY (cari_id, stock_id)
                )
                """
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_next_date "
                f"ON {table_sql} (tahmini_siparis_tarihi)"
            )
        )
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_status ON {table_sql} (tahmin_durumu)"))
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_confidence ON {table_sql} (guven_seviyesi)"))
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_cari_kod ON {table_sql} (cari_kod)"))
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_stock_kod ON {table_sql} (stock_kod)"))


def prepare_table(engine, table: str, schema: str | None, table_kind: str, if_exists: str) -> None:
    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("--if-exists must be one of: fail, replace, append")

    table_creator = {
        "historical": create_historical_sales_table,
        "predictions": create_next_order_predictions_table,
    }[table_kind]

    create_schema_if_needed(engine, schema)
    if if_exists == "replace":
        recreate_table(engine, table, schema)
        table_creator(engine, table, schema)
    else:
        table_creator(engine, table, schema)
        if if_exists == "fail":
            table_sql = table_reference(table, schema)
            with engine.begin() as connection:
                count = connection.execute(text(f"SELECT COUNT(*) FROM {table_sql}")).scalar_one()
            if count:
                raise ValueError(f"{table_sql} already has {count:,} rows. Use --if-exists replace or append.")
