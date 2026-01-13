import yfinance as yf

# Download S&P 500 index data
data = yf.download("^GSPC", start="2023-01-01", end="2026-01-01", auto_adjust=True)

print(data.head())
data.to_parquet("SP500_index.parquet")
