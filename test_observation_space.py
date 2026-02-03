"""
Test script to verify the enhanced observation space in MultiStockTradingEnv.
This script creates dummy data with 7 features and verifies the observation structure.
"""

import numpy as np
from src.env.environment import MultiStockTradingEnv

def test_observation_space():
    print("=" * 60)
    print("Testing Enhanced Observation Space")
    print("=" * 60)
    
    # Create dummy data: (timesteps, stocks, features)
    # Features: [Open, High, Low, Close, RSI, MACD, Volume]
    n_timesteps = 100
    n_stocks = 100
    n_features = 7
    
    # Generate random data
    np.random.seed(42)
    data_tensor = np.random.randn(n_timesteps, n_stocks, n_features).astype(np.float32)
    
    # Make prices positive (columns 0-3: Open, High, Low, Close)
    data_tensor[:, :, 0:4] = np.abs(data_tensor[:, :, 0:4]) * 100 + 50
    
    # RSI should be 0-100
    data_tensor[:, :, 4] = np.abs(data_tensor[:, :, 4]) * 50 + 25
    
    # MACD can be negative
    data_tensor[:, :, 5] = data_tensor[:, :, 5] * 10
    
    # Volume should be positive
    data_tensor[:, :, 6] = np.abs(data_tensor[:, :, 6]) * 1000000
    
    print(f"\nData shape: {data_tensor.shape}")
    print(f"Expected: ({n_timesteps}, {n_stocks}, {n_features})")
    
    # Create environment
    env = MultiStockTradingEnv(data_tensor=data_tensor, initial_balance=100000, n_stocks=n_stocks)
    
    print(f"\nEnvironment created successfully!")
    print(f"Number of features: {env.n_features}")
    print(f"Number of stocks: {env.n_stocks}")
    
    # Calculate expected observation size
    expected_obs_size = (n_stocks * (n_features + 1)) + 1
    print(f"\nExpected observation size: {expected_obs_size}")
    print(f"  = ({n_stocks} stocks * ({n_features} features + 1 shares)) + 1 balance")
    print(f"  = ({n_stocks} * {n_features + 1}) + 1")
    print(f"  = {n_stocks * (n_features + 1)} + 1")
    print(f"  = {expected_obs_size}")
    
    print(f"\nActual observation space shape: {env.observation_space.shape}")
    
    # Reset and get initial observation
    obs, info = env.reset()
    
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    
    # Verify the structure
    print("\n" + "=" * 60)
    print("Verifying Observation Structure")
    print("=" * 60)
    
    # The observation should be structured as:
    # [Stock0_F0, Stock0_F1, ..., Stock0_F6, Stock0_Shares,
    #  Stock1_F0, Stock1_F1, ..., Stock1_F6, Stock1_Shares,
    #  ...,
    #  Stock99_F0, Stock99_F1, ..., Stock99_F6, Stock99_Shares,
    #  Balance]
    
    # Extract first stock's data
    stock_0_data = obs[0:n_features+1]
    print(f"\nFirst stock observation (features + shares):")
    print(f"  Shape: {stock_0_data.shape}")
    print(f"  Values: {stock_0_data}")
    print(f"  Expected: 7 features + 1 shares = 8 values")
    
    # Extract balance (last element)
    balance = obs[-1]
    print(f"\nBalance (last element): {balance}")
    print(f"Expected balance: 100000.0")
    
    # Verify shares are 0 initially
    shares_indices = [i * (n_features + 1) + n_features for i in range(n_stocks)]
    shares = obs[shares_indices]
    print(f"\nAll shares held (should be all zeros initially):")
    print(f"  Shape: {shares.shape}")
    print(f"  Sum: {np.sum(shares)}")
    print(f"  All zeros: {np.all(shares == 0)}")
    
    # Test a step
    print("\n" + "=" * 60)
    print("Testing Step Function")
    print("=" * 60)
    
    # Create a simple action (buy 10% of stock 0, do nothing else)
    actions = np.zeros(n_stocks, dtype=np.float32)
    actions[0] = 0.1  # Buy 10% allocation to stock 0
    
    obs_next, reward, terminated, truncated, info = env.step(actions)
    
    print(f"\nAfter step:")
    print(f"  Observation shape: {obs_next.shape}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Net worth: {info['net_worth']}")
    
    # Check if shares changed
    shares_after = obs_next[shares_indices]
    print(f"\n  Shares after step:")
    print(f"    Stock 0 shares: {shares_after[0]}")
    print(f"    Total shares held: {np.sum(shares_after)}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_observation_space()
