import os
import numpy as np
import pandas as pd
import gymnasium as gym
from ta.momentum import RSIIndicator
from ta.trend import MACD
import yfinance as yf
# --- Imports from your project structure ---
from src.env.environment import MultiStockTradingEnv  # Updated class name
from src.ppo.PPO_Classes import Agent
from src.ppo.utils import plot_learning_curve

# --- 1. Helper: Synthetic Data Generator ---
def make_test_df(ticker):
    # 1) Download AAPL daily data for the past 2 years
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # yfinance returns DatetimeIndex; keep it if you want, but env doesn't require it
    df = df.reset_index()

    # 2) Standardize column names (sometimes yfinance provides MultiIndex columns)
    # If you get a MultiIndex like ('Open','AAPL'), flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # 3) Ensure required OHLC columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from yfinance: {missing}. Got: {df.columns.tolist()}")

    # 4) Compute indicators required by your env: RSI and MACD (based on Close)
    close = df["Close"].astype(float)

    df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    df["MACD"] = MACD(close=close, window_slow=26, window_fast=12, window_sign=9).macd()

    # 5) Drop indicator warmup NaNs and reset index
    df = df.dropna(subset=["RSI", "MACD"]).reset_index(drop=True)

    # 6) Keep only what the env needs (optional, but clean)
    df = df[["Open", "High", "Low", "Close", "RSI", "MACD", "Volume"]].copy()

    # 7) Enforce numeric dtypes
    for col in df.columns:
        df[col] = df[col].astype(np.float32)

    return df

# --- 2. Helper: simple Normalization Wrapper ---
# PPO fails if inputs are 400.0 and 0.001 mixed together.
class SimpleNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def normalize(self, obs):
        # Observation structure: [Stock0_Data(7)+Shares(1), Stock1..., Balance(1)]
        # Total per stock = 8
        obs = np.array(obs, dtype=np.float32)
        
        n_features_plus_shares = 8
        n_stocks = (len(obs) - 1) // n_features_plus_shares
        
        # Normalize stock data
        for i in range(n_stocks):
            base = i * n_features_plus_shares
            
            # Prices: Open, High, Low, Close (Indices 0-3)
            obs[base:base+4] /= 1000.0
            
            # RSI (Index 4)
            obs[base+4] /= 100.0
            
            # MACD (Index 5)
            obs[base+5] /= 20.0
            
            # Volume (Index 6) - Normalize by something large, e.g., 10M or 100M
            obs[base+6] /= 1_000_000.0
            
            # Shares (Index 7)
            obs[base+7] /= 100.0
            
        # Normalize Balance (Last element)
        obs[-1] /= 100000.0
        
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info

# --- 3. Main Training Loop ---
# 1. Create Data and Environment
mock_data_path = "data/mock_market_data.npy"

if os.path.exists(mock_data_path):
    print(f"Loading mock data from {mock_data_path}...")
    data_tensor = np.load(mock_data_path)
    # data_tensor shape should be (n_timesteps, n_stocks, n_features)
    n_stocks_loaded = data_tensor.shape[1]
    print(f"Loaded data for {n_stocks_loaded} stocks, {data_tensor.shape[0]} timesteps.")
    
    print(f"Loaded data for {n_stocks_loaded} stocks, {data_tensor.shape[0]} timesteps.")
    
    # Switch to DSR for advanced optimization
    raw_env = MultiStockTradingEnv(data_tensor=data_tensor, n_stocks=n_stocks_loaded, reward_type="dsr")
else:
    print("Mock data not found. downloading TSLA data...")
    df = make_test_df("TSLA")
    # Expand df dims to match (Time, Stocks, Features) expected by Env
    # We'll duplicate the single stock data 100 times to simulate 100 stocks for testing
    # Data: (Time, Features) -> (Time, 1, Features) -> (Time, 100, Features)
    data_tensor = np.expand_dims(df.values, axis=1) # (Time, 1, 7)
    data_tensor = np.tile(data_tensor, (1, 100, 1)) # (Time, 100, 7)

    data_tensor = np.tile(data_tensor, (1, 100, 1)) # (Time, 100, 7)

    # Switch to DSR for advanced optimization
    raw_env = MultiStockTradingEnv(data_tensor=data_tensor, n_stocks=100, reward_type="dsr")

# 2. Wrap environment to normalize inputs (CRITICAL FOR PPO)
env = SimpleNormalizeWrapper(raw_env)

# 3. Hyperparameters
N = 2048
batch_size = 64
n_epochs = 10
alpha = 0.0003
n_games = 1500

agent = Agent(
    n_actions=env.action_space.shape[0], # Use shape[0] for continuous action vector size
    batch_size=batch_size,
    alpha=alpha,
    n_epochs=n_epochs,
    input_dims=env.observation_space.shape,
)

# Load existing models if they exist (Continual Learning)
if os.path.exists("actor_torch_ppo.pth") and os.path.exists("critic_torch_ppo.pth"):
    print("Found existing models, loading...")
    agent.load_models()
else:
    print("No existing models found, starting from scratch.")

figure_file = "trading_agent_learning_curve.png"
figure_file2 = "trading_agent_balance.png"
best_score = -np.inf
score_history = []
price_history = []
learn_iters = 0
avg_score = 0.0
n_steps = 0

print("Starting training...")

for i in range(n_games):
    observation, info = env.reset()
    terminated = False
    truncated = False
    score = 0.0

    while not (terminated or truncated):
        action, prob, val = agent.choose_action(observation)

        action_env = action # Steps are fully compatible now

        observation_, reward, terminated, truncated, info = env.step(action_env)
        info = info['net_worth']
        n_steps += 1
        score += reward

        done_flag = terminated or truncated
        agent.remember(observation, action, prob, val, reward, done_flag)

        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1

        observation = observation_

    score_history.append(score)
    price_history.append(info)
    avg_price = np.mean(price_history[-100:])
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print(
        f"Episode {i} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Steps: {n_steps} | Balance: {info} |Avg Balance {avg_price}")

# Plotting
x = [i + 1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)
plot_learning_curve(x, price_history, figure_file2)