import numpy as np
import pandas as pd
import gymnasium as gym
from ta.momentum import RSIIndicator
from ta.trend import MACD
import yfinance as yf
import time
import torch as T
# --- Imports from your project structure ---
from src.env.environment import SP500TradingEnv  # Corrected import name
from src.ppo.PPO_Classes import Agent
from src.ppo.utils import plot_learning_curve

print(f"CUDA Available: {T.cuda.is_available()}")
print(f"PyTorch Version: {T.__version__}")
print(f"CUDA Version in PyTorch: {T.version.cuda}")

T.set_float32_matmul_precision('high')


# --- 1. Helper: Synthetic Data Generator ---
def make_test_df(ticker):
    # 1) Download daily data for the past 2 years
    df = yf.download(
        ticker,
        period="730d",  # 2 years of data
        interval="1h",  # DAILY data for faster training (not hourly)
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
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from yfinance: {missing}. Got: {df.columns.tolist()}")

    # 4) Compute indicators required by your env: RSI and MACD (based on Close)
    close = df["Close"].astype(float)

    df["RSI"] = RSIIndicator(close=close, window=14 * 6).rsi()
    df["MACD"] = MACD(close=close, window_slow=26 * 6, window_fast=12 * 6, window_sign=9 * 6).macd()

    # 5) Drop indicator warmup NaNs and reset index
    df = df.dropna(subset=["RSI", "MACD"]).reset_index(drop=True)

    # 6) Keep only what the env needs (optional, but clean)
    df = df[["Open", "High", "Low", "Close", "RSI", "MACD"]].copy()

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
        # A simple manual scaling strategy
        # [Open, High, Low, Close, RSI, MACD, Balance, Shares]
        # We scale prices by 1000, RSI by 100, Balance by initial_balance scale
        obs = np.array(obs, dtype=np.float32)
        obs[0:4] /= 1000.0  # Scale Prices
        obs[4] /= 100.0  # Scale RSI
        obs[5] /= 20.0  # Scale MACD (Approx range -20 to 20 for high priced stocks)
        obs[6] /= 100000.0  # Scale Balance (matches initial balance order of magnitude)
        obs[7] /= 1000.0  # Scale Shares (Allow for higher share counts)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info


# --- 3. Main Training Loop ---
if __name__ == "__main__":
    # 1. Create Data and Environment
    print("Downloading data...")
    df = make_test_df("TSLA")
    print(f"Data loaded: {len(df)} days")

    raw_env = SP500TradingEnv(df=df)

    # 2. Wrap environment to normalize inputs (CRITICAL FOR PPO)
    env = SimpleNormalizeWrapper(raw_env)

    # 3. Hyperparameters - OPTIMIZED FOR GPU UTILIZATION
    N = 64  # Learn every 64 steps (was 200) - more frequent GPU usage
    batch_size = 64  # Much larger batches (was 10) - better GPU utilization
    n_epochs = 10  # More training iterations (was 4) - more GPU work per learn()
    alpha = 0.0003  # Learning rate
    n_games = 1500  # Total episodes

    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    figure_file = "trading_agent_learning_curve.png"
    figure_file2 = "trading_agent_balance.png"
    best_score = -np.inf
    score_history = []
    price_history = []
    learn_iters = 0
    avg_score = 0.0
    n_steps = 0

    print("Starting training...")
    print(f"Hyperparameters: N={N}, batch_size={batch_size}, n_epochs={n_epochs}, alpha={alpha}")

    for i in range(n_games):
        observation, info = env.reset()
        terminated = False
        truncated = False
        score = 0.0

        while not (terminated or truncated):
            # Choose action
            action, prob, val = agent.choose_action(observation)

            # Convert Tensor/Array action to standard Python Int for the Env
            if hasattr(action, 'item'):
                action_env = action.item()
            else:
                action_env = action

            observation_, reward, terminated, truncated, info = env.step(action_env)
            net_worth = info['net_worth']
            n_steps += 1
            score += reward

            done_flag = terminated or truncated
            agent.remember(observation, action, prob, val, reward, done_flag)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_

        score_history.append(score)
        price_history.append(net_worth)
        avg_price = np.mean(price_history[-100:])
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            f"Episode {i} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | "
            f"Steps: {n_steps} | Balance: {net_worth:.2f} | Avg Balance: {avg_price:.2f}")

    # Plotting
    print("Generating plots...")
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    plot_learning_curve(x, price_history, figure_file2)
    print(f"Training complete! Plots saved to {figure_file} and {figure_file2}")