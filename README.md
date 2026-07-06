# Customer Next Order Prediction

Machine Learning project using Pandas, Scikit-learn and PostgreSQL.

## Amaç

Bu proje haftalık geçmiş satış verilerinden her cari-ürün ikilisi için bir sonraki sipariş tarihini ve tahmini miktarı üretir.

Girdi Excel kolonları:

- `CariId`, `CariKod`
- `StockId`, `StockKod`, `StockAd`
- `Hafta`
- `ToplamMiktar`, `ToplamTutar`, `SiparisSatirSayisi`

## Klasörler

- `data/raw`: Ham Excel dosyaları. GitHub'a gönderilmez.
- `data/processed`: İşlenmiş ara veriler. GitHub'a gönderilmez.
- `src`: Python kaynak kodları.
- `models`: Eğitilmiş model çıktıları. GitHub'a gönderilmez.
- `outputs`: Tahmin Excel çıktıları. GitHub'a gönderilmez.

## Doğru Ana Akış

İstenen üretim sırası budur:

1. Excel dosyasını oku.
2. PostgreSQL'de `historical_sales_weekly` satış tablosunu oluştur.
3. Excel verisini bu tabloya yükle.
4. PostgreSQL'den satış verisini tekrar oku.
5. Machine learning modeliyle tahmin üret.
6. PostgreSQL'de `next_order_predictions` tahmin tablosunu oluştur.
7. Tahminleri PostgreSQL tahmin tablosuna yaz.
8. Yeni Excel dosyasını PostgreSQL tahmin tablosundan export et.

Tek komut:

```powershell
.\.venv\Scripts\python.exe -m src.run_postgres_pipeline
```

Varsayılan yeni Excel çıktısı tek sayfadır ve sadece 7 kolon içerir:

```text
outputs/next_order_predictions_from_postgres.xlsx
```

Kolonlar:

- `CariId`
- `CariKod`
- `StockId`
- `StockKod`
- `StockAd`
- `TahminiSiparisTarihi`
- `TahminiMiktar`

Bu bilgisayarda proje için local PostgreSQL cluster `pgdata/` altında oluşturuldu ve `.env` dosyası `localhost:5433` portunu kullanıyor.

PostgreSQL kapalıysa tekrar başlatmak:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D pgdata -l logs\postgresql-local.log -o "-p 5433" start
```

PostgreSQL'i durdurmak:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe" -D pgdata stop
```

Tablo satırlarını kontrol etmek:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U postgres -d customer_next_order -c "SELECT COUNT(*) FROM historical_sales_weekly;"
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -p 5433 -U postgres -d customer_next_order -c "SELECT COUNT(*) FROM next_order_predictions;"
```

## PostgreSQL'e Sadece Satış Verisi Yükleme

Önce `.env.example` içindeki PostgreSQL bilgilerini kendi ortamına göre `.env` veya terminal environment variable olarak ayarla.

Dry-run ile veri kontrolü:

```powershell
.\.venv\Scripts\python.exe -m src.load_to_postgres --dry-run
```

Gerçek yükleme:

```powershell
.\.venv\Scripts\python.exe -m src.load_to_postgres --if-exists replace
```

Varsayılan tablo adı: `historical_sales_weekly`

## Sadece Tahmin Üretme

Excel dosyasından model eğitip tahmin Excel'i üretmek:

```powershell
.\.venv\Scripts\python.exe -m src.train_predict_next_order --source excel
```

PostgreSQL tablosundan okuyup tahmin üretmek:

```powershell
.\.venv\Scripts\python.exe -m src.train_predict_next_order --source postgres
```

PostgreSQL'den okuyup tahminleri ayrıca PostgreSQL tablosuna yazmak:

```powershell
.\.venv\Scripts\python.exe -m src.train_predict_next_order --source postgres --write-postgres
```

Varsayılan tahmin çıktısı:

```text
outputs/next_order_predictions.xlsx
```

Excel içindeki sayfa:

- `tahminler`: Tüm cari-ürün tahminleri.
