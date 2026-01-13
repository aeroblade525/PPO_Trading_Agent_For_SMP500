import pandas as pd
import yfinance as yf

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

tables = pd.read_html(url)
sp500 = tables[0]             
tickers = sp500["Symbol"].tolist()
tickers = [t.replace(".", "-") for t in tickers]

data = yf.download(
    tickers,
    start="2023-01-01",
    end="2026-01-01",
    auto_adjust=True,  
    group_by="ticker",
    threads=True
)

data.to_parquet("SP500_prices.parquet")
print(data.head())
