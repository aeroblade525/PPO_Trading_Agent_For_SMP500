from env.environment import SP500TradingEnv as gym
import numpy as np
from ppo.PPO_Classes import Agent
from ppo.utils import plot_learning_curve
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

def make_test_df(n=300, seed=42, start_price=400.0):
    rng = np.random.default_rng(seed)

    # Synthetic close series (random walk)
    rets = rng.normal(loc=0.0002, scale=0.01, size=n)  # small drift
    close = start_price * np.exp(np.cumsum(rets))

    # Create plausible OHLC around close
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low  = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=n)))

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
    })

    # Indicators
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD"] = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9).macd()

    # Drop initial NaNs from indicator warmup
    df = df.dropna().reset_index(drop=True)
    return df

# Example usage:
test_df = make_test_df(n=400)
print(test_df.head())
print(test_df.columns)

if __name__ == "__main__":
    env = gym(make_test_df())

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    n_games = 300
    figure_file = "cartpole.png"

    best_score = -np.inf
    score_history = []

    learn_iters = 0
    avg_score = 0.0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        terminated = False
        truncated = False
        score = 0.0

        while not (terminated or truncated):
            action, prob, val = agent.choose_action(observation)

            observation_, reward, terminated, truncated, info = env.step(action)

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

        print(
            f"episode {i} score {round(score, 1)} avg score {round(avg_score, 1)} "
            f"time_steps {n_steps} learning_steps {learn_iters}"
        )

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)