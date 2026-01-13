import numpy as np
import pandas as pd
import gymnasium as gym
from ta.momentum import RSIIndicator
from ta.trend import MACD

# --- Imports from your project structure ---
from src.env.environment import SP500TradingEnv  # Corrected import name
from src.ppo.PPO_Classes import Agent
from src.ppo.utils import plot_learning_curve

# --- 1. Helper: Synthetic Data Generator ---
def make_test_df(n=300, seed=42, start_price=400.0):
    rng = np.random.default_rng(seed)
    
    # Synthetic random walk
    rets = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = start_price * np.exp(np.cumsum(rets))
    
    # Create plausible OHLC
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low  = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
    })
    
    # Add Indicators
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD"] = MACD(close=df["Close"], window_slow=26, window_fast=12).macd()
    
    # Drop NaNs
    df = df.dropna().reset_index(drop=True)
    return df

# --- 2. Helper: simple Normalization Wrapper ---
# PPO fails if inputs are 400.0 and 0.001 mixed together.
class SimpleNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def normalize(self, obs):
        # A simple manual scaling strategy
        # [Open, High, Low, Close, RSI, MACD, Balance, Shares]
        # We scale prices by 1000, RSI by 100, Balance by 10000
        obs = np.array(obs, dtype=np.float32)
        obs[0:4] /= 1000.0  # Scale Prices
        obs[4] /= 100.0     # Scale RSI
        obs[6] /= 10000.0   # Scale Balance
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
    df = make_test_df(n=500)
    raw_env = SP500TradingEnv(df=df)
    
    # 2. Wrap environment to normalize inputs (CRITICAL FOR PPO)
    env = SimpleNormalizeWrapper(raw_env)

    # 3. Hyperparameters
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    n_games = 300
    
    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    figure_file = "trading_agent_learning_curve.png"
    best_score = -np.inf
    score_history = []
    
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
            # FIX: Ensure proper shape/type for custom agents
            action, prob, val = agent.choose_action(observation)
            
            # FIX: Convert Tensor/Array action to standard Python Int for the Env
            if hasattr(action, 'item'):
                action_env = action.item() 
            else:
                action_env = action

            observation_, reward, terminated, truncated, info = env.step(action_env)
            
            n_steps += 1
            score += reward
            
            done_flag = terminated or truncated
            agent.remember(observation, action, prob, val, reward, done_flag)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"Episode {i} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Steps: {n_steps}")

    # Plotting
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)