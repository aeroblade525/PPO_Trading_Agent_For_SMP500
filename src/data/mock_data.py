import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os

def generate_stock_data(n_timesteps=1000, start_price=100.0, volatility=0.2, drift=0.05):
    """
    Generates synthetic stock data using Geometric Brownian Motion.
    """
    dt = 1/252 # Daily steps
    
    # Generate random path (returns)
    # r = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
    returns = np.random.normal(loc=(drift - 0.5 * volatility**2) * dt, 
                               scale=volatility * np.sqrt(dt), 
                               size=n_timesteps)
    
    # Price path
    price_path = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from Close path
    close = price_path
    
    # Open is previous Close
    open_prices = np.roll(close, 1)
    # First Open is start_price
    open_prices[0] = start_price
    
    # Add noise to High and Low
    daily_volatility = price_path * volatility * np.sqrt(dt) 
    high = np.maximum(open_prices, close) + np.abs(np.random.normal(0, daily_volatility, n_timesteps))
    low = np.minimum(open_prices, close) - np.abs(np.random.normal(0, daily_volatility, n_timesteps))
    
    # Volume: Log normal distribution
    # Mean ~ 10M
    volume = np.random.lognormal(mean=16.1, sigma=0.5, size=n_timesteps) 
    
    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })
    
    return df

def add_indicators(df):
    """
    Adds RSI and MACD indicators.
    """
    close = df["Close"]
    
    # RSI
    df["RSI"] = RSIIndicator(close=close, window=14).rsi().fillna(50)
    
    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd().fillna(0)
    
    return df

def generate_market_tensor(n_stocks=100, n_timesteps=2000):
    """
    Generates a 3D tensor of market data for multiple stocks.
    Shape: (n_timesteps, n_stocks, n_features)
    Features: [Open, High, Low, Close, RSI, MACD, Volume]
    """
    all_data = []
    
    print(f"Generating data for {n_stocks} stocks over {n_timesteps} steps...")
    
    for i in range(n_stocks):
        # Randomize parameters for each stock
        start_price = np.random.uniform(20, 500)
        volatility = np.random.uniform(0.1, 0.5) # 10% to 50% annual volatility
        drift = np.random.uniform(-0.1, 0.2)     # -10% to +20% annual drift
        
        df = generate_stock_data(n_timesteps, start_price, volatility, drift)
        df = add_indicators(df)
        
        # Select features
        features = df[["Open", "High", "Low", "Close", "RSI", "MACD", "Volume"]].values
        all_data.append(features)
        
    # Stack along axis 1
    # feature list is (n_stocks, n_timesteps, n_features)
    # We want (n_timesteps, n_stocks, n_features)
    data = np.stack(all_data, axis=1) # (T, N, F)
    
    return data.astype(np.float32)

if __name__ == "__main__":
    # Create output directory if needed
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "mock_market_data.npy")
    
    # Generate
    data = generate_market_tensor(n_stocks=100, n_timesteps=5000)
    
    # Save
    np.save(output_path, data)
    print(f"Saved mock data to {os.path.abspath(output_path)}")
    print(f"Data Shape: {data.shape}")
