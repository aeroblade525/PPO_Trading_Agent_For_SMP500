import pandas as pd
import yfinance as yf
import sqlite3
from pathlib import Path


# =========================
# CONFIG
# =========================
DB_PATH = "src/data/sp500.sqlite"   # use .sqlite if you want
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"

BLACKLIST = {"GEV", "KVUE", "SOLV", "VLTO", "WBA"}
TICKER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"


# =========================
# SQLITE INIT
# =========================
def init_sqlite_db(db_path: str):
    """
    Initialize SQLite database and schema.
    Safe to call multiple times.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date   TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (date, ticker)
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_prices_date
        ON prices(date);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_prices_ticker
        ON prices(ticker);
    """)

    conn.commit()
    conn.close()


# =========================
# TICKERS
# =========================
def get_sp500_tickers():
    df = pd.read_csv(TICKER_URL)
    tickers = df["Symbol"].tolist()

    # Yahoo uses hyphens instead of dots
    tickers = [t.replace(".", "-") for t in tickers if t not in BLACKLIST]

    print(f"Loaded {len(tickers)} S&P 500 tickers")
    return tickers


# =========================
# DOWNLOAD
# =========================
def download_yahoo_data(tickers):
    print("Downloading Yahoo Finance data...")

    df = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True
    )

    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned empty data")

    return df


# =========================
# TRANSFORM
# =========================
def to_long_format(df):
    """
    Convert MultiIndex columns:
    (Ticker, Feature) → long format
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance (multi-ticker download).")

    # Pandas warning fix: future_stack=True
    long_df = (
        df.stack(level=0, future_stack=True)
          .rename_axis(index=["Date", "Ticker"])
          .reset_index()
    )

    # Enforce numeric dtypes (avoid object garbage)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    # Drop rows where price is missing (common for newly added tickers)
    long_df = long_df.dropna(subset=["Close"])

    return long_df


# =========================
# INSERT
# =========================
def insert_prices(df, db_path: str, clear_table: bool = True):
    """
    Insert into SQLite without hitting 'too many SQL variables' by chunking.
    Optionally clears table first (recommended for re-runs).
    """
    conn = sqlite3.connect(db_path)

    if clear_table:
        conn.execute("DELETE FROM prices;")
        conn.commit()

    df = df.rename(columns={
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Ensure date stored cleanly as text YYYY-MM-DD
    df["date"] = df["date"].astype(str)

    # CRITICAL FIX:
    # - chunksize keeps each insert small enough for SQLite variable limits
    # - method=None avoids multi-row single statement that explodes variable count
    df.to_sql(
        "prices",
        conn,
        if_exists="append",
        index=False,
        chunksize=200,   # safe; increase cautiously if you want speed
        method=None
    )

    conn.close()


# =========================
# MAIN
# =========================
def main():
    init_sqlite_db(DB_PATH)

    tickers = get_sp500_tickers()
    raw_df = download_yahoo_data(tickers)
    long_df = to_long_format(raw_df)

    insert_prices(long_df, DB_PATH, clear_table=True)

    print("\nSUCCESS ✅")
    print(f"Rows inserted: {len(long_df)}")
    print(long_df.head())


if __name__ == "__main__":
    main()