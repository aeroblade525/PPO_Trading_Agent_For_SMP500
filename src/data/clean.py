import pandas as pd
import yfinance as yf
import os

def get_sp500_data():
    # 1. Tickers to ignore (ones that cause errors or don't exist in your timeframe)
    blacklist = ['GEV', 'KVUE', 'SOLV', 'VLTO', 'WBA']
    
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    
    print("Fetching ticker list...")
    try:
        ticker_df = pd.read_csv(url)
        col_name = 'Symbol' if 'Symbol' in ticker_df.columns else ticker_df.columns[0]
        
        # Get raw list
        raw_tickers = ticker_df[col_name].tolist()
        
        # Filter out blacklisted tickers and clean dots to hyphens
        tickers = [
            t.replace('.', '-') for t in raw_tickers 
            if t not in blacklist
        ]
        
    except Exception as e:
        print(f"Failed to fetch online list: {e}")
        tickers = ['AAPL', 'MSFT', 'NVDA'] # Minimal fallback

    print(f"Downloading data for {len(tickers)} tickers (excluding {len(blacklist)} blacklisted)...")

    # 2. Download from yfinance
    data = yf.download(
        tickers=tickers,
        start="2023-01-01",
        end="2024-01-01", # Extended range for better PPO training
        group_by='column',
        threads=True
    )

    return data

if __name__ == "__main__":
    df = get_sp500_data()
    
    if df is not None and not df.empty:
        # Save to your project folder
        output_path = os.path.join('src', 'data', 'sp500_individual_stocks.csv')
        df.to_csv(output_path)
        print(f"\nSuccess! Cleaned data saved to {output_path}")
    else:
        print("Download failed.")