import pandas as pd
import yfinance as yf
import sqlite3
from pathlib import Path

# CONFIG
DB_PATH = "src/data/sp100.sqlite"
START_DATE = "2023-01-01"
END_DATE = "2026-01-01"

# The "All-Star" 100 companies (Sub-set of S&P 500)
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", 
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", 
    "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", 
    "CVS", "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", 
    "META", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", 
    "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", 
    "LIN", "LLY", "LMT", "LOW", "LRCX", "MA", "MCD", "MDLZ", "MDT", "MET", 
    "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", 
    "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", 
    "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", 
    "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"
]

def main():
    # 1. Setup DB
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Download (threads=True makes this very fast)
    print(f"Downloading {len(SP100_TICKERS)} S&P 100 tickers...")
    raw_df = yf.download(
        tickers=SP100_TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True
    )

    # 3. Transform to Long Format
    long_df = raw_df.stack(level=0, future_stack=True).reset_index()
    long_df.columns = [c.lower() for c in long_df.columns]
    
    # Standardize column names
    if 'level_1' in long_df.columns:
        long_df.rename(columns={'level_1': 'ticker'}, inplace=True)
    
    long_df['date'] = long_df['date'].dt.strftime('%Y-%m-%d')
    long_df = long_df.dropna(subset=['close'])

    # 4. Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    long_df.to_sql("prices", conn, if_exists="replace", index=False)
    
    # Create an index to make your Trading Agent faster
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON prices (ticker, date);")
    conn.close()
    
    print(f"SUCCESS: {len(long_df)} rows saved to {DB_PATH}")

if __name__ == "__main__":
    main()