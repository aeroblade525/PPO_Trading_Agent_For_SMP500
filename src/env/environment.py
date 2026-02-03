import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_tensor, initial_balance=100000, n_stocks=100, reward_type="simple"):
        super(MultiStockTradingEnv, self).__init__()
        
        # Features expected: [Open, High, Low, Close, RSI, MACD, Volume]
        self.data = data_tensor
        self.n_stocks = n_stocks
        self.n_features = data_tensor.shape[2]
        self.initial_balance = initial_balance
        self.reward_type = reward_type # Options: "simple", "dsr"
        
        # Windowing (Train on chunks of data rather than the full history every time)
        self.window_size = 500  # Default to ~2 years of data per episode
        self.total_timesteps = data_tensor.shape[0]
        self.start_step = 0
        self.end_step = self.total_timesteps
        
        # --- ACTION SPACE ---
        # A vector of 100 continuous numbers between -1 and 1
        # Positive = Buy weight, Negative = Sell percentage
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        # --- OBSERVATION SPACE ---
        # Interleaved structure: For each stock, we have (n_features + 1) values:
        # [Stock1_Feature1, ..., Stock1_FeatureN, Stock1_Shares, Stock2_Feature1, ...] + Balance
        # Size = (n_stocks * (n_features + 1)) + 1 balance
        obs_size = (self.n_stocks * (self.n_features + 1)) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Internal State
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        
        # DSR Variables
        self.eta = 1 / 252
        self.A = 0.0
        self.B = 0.0
        self.prev_net_worth = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random start point for this episode
        # Ensure we have at least window_size remaining
        max_start = self.total_timesteps - self.window_size
        if max_start > 0:
            self.start_step = np.random.randint(0, max_start)
        else:
            self.start_step = 0
            
        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.window_size, self.total_timesteps - 1)
        
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float32)
        self.prev_net_worth = self.initial_balance
        self.A = 0.0
        self.B = 0.0
        
        return self._next_observation(), {}

    def step(self, actions):
        # 1. Get Current Prices for all 100 stocks
        # Assuming 'Close' is at index 3 in our features
        current_prices = self.data[self.current_step, :, 3] 
        
        # 2. Execute Trades
        action_penalty = self._take_action(actions, current_prices)
        
        # 3. Move Time Forward
        self.current_step += 1
        
        # 4. Check Termination
        # 4. Check Termination
        # Terminate if we reach the end of the window OR the end of data
        if self.current_step >= self.end_step:
            return np.zeros(self.observation_space.shape), 0, True, False, {"net_worth": self.prev_net_worth}

        # 5. Calculate Portfolio Value
        # Next Step Prices (to calculate immediate PnL)
        next_prices = self.data[self.current_step, :, 3]
        
        # Vectorized Value Calculation: Cash + (Vector of Shares * Vector of Prices)
        stock_value = np.sum(self.shares_held * next_prices)
        total_net_worth = self.balance + stock_value
        
        # 6. Reward Calculation
        if self.reward_type == "dsr":
            reward = self._calculate_robust_dsr(total_net_worth, self.prev_net_worth)
        else:
            # Default to simple return
            reward = self._calculate_reward(total_net_worth, self.prev_net_worth)
        
        # Apply penalty (already negative)
        reward += action_penalty 
        
        self.prev_net_worth = total_net_worth
        
        # Bankruptcy Check
        terminated = total_net_worth <= 0
        if terminated: reward = -10.0

        return self._next_observation(), reward, terminated, False, {"net_worth": total_net_worth}

    def _take_action(self, actions, current_prices):
        # actions is a vector of 100 floats (-1 to 1)
        action_penalty = 0.0
        
        # --- PHASE 1: SELLING (Vectorized) ---
        # Find indices where we want to sell (action < 0)
        sell_indices = np.where(actions < 0)[0]
        
        if len(sell_indices) > 0:
            # Calculate sell fractions: clip(abs(action), 0, 1)
            sell_fractions = np.clip(np.abs(actions[sell_indices]), 0.0, 1.0)
            
            # Calculate amount to sell
            shares_held_to_sell = self.shares_held[sell_indices]
            shares_to_sell = shares_held_to_sell * sell_fractions
            
            # Update holdings
            self.shares_held[sell_indices] -= shares_to_sell
            
            # Update balance (revenue)
            revenue = np.sum(shares_to_sell * current_prices[sell_indices])
            self.balance += revenue

        # --- PHASE 2: BUYING (Vectorized) ---
        buy_indices = np.where(actions > 0)[0]
        
        if len(buy_indices) > 0:
            buy_weights = actions[buy_indices]
            
            # Normalize weights
            total_weight = np.sum(buy_weights)
            scale = 1.0
            
            if total_weight > 1.0:
                # Penalty removed to allow agent to focus on returns.
                # The constraints are enforced by scaling anyway.
                action_penalty = 0.0
                scale = 1.0 / total_weight
                
            # Allocate cash: balance * (weight * scale)
            # vector of shape (len(buy_indices),)
            allocations = self.balance * (buy_weights * scale)
            
            # Get prices for buying stocks
            buy_prices = current_prices[buy_indices]
            
            # Avoid division by zero
            valid_price_mask = buy_prices > 0
            
            if np.any(valid_price_mask):
                valid_allocations = allocations[valid_price_mask]
                valid_prices = buy_prices[valid_price_mask]
                valid_indices = buy_indices[valid_price_mask]
                
                # Calculate whole shares
                shares_to_buy = np.floor(valid_allocations / valid_prices)
                
                # Update state
                cost = np.sum(shares_to_buy * valid_prices)
                self.shares_held[valid_indices] += shares_to_buy
                self.balance -= cost
        return action_penalty

    def _calculate_reward(self, current_net_worth, prev_net_worth):
        """
        Calculate simple percentage return for stable training.
        """
        if prev_net_worth == 0:
            return 0.0
            
        # Simple Return
        pct_change = (current_net_worth - prev_net_worth) / prev_net_worth
        
        # Scale up (e.g. 0.01% -> 0.0001 -> 1.0 reward unit?) 
        # Returns act are roughly +/- 1-5% per day? NO, per step (day) it is usually +/- 0.1% to 2%
        # If return is 1% (0.01), let's make that reward 1.0
        return pct_change * 100.0

    def _calculate_robust_dsr(self, current_net_worth, prev_net_worth):
        """
        Calculate a ROBUST Differential Sharpe Ratio (DSR).
        - Prevents division by zero.
        - Clips output to reasonable range [-5, 5].
        """
        # 1. Calculate Return
        if prev_net_worth == 0: 
            return 0.0
        r_t = (current_net_worth - prev_net_worth) / prev_net_worth
        
        # 2. Update Running Stats (Exponential Moving Average)
        self.A = self.A + self.eta * (r_t - self.A)
        self.B = self.B + self.eta * (r_t**2 - self.B)
        
        # 3. Calculate Variance with Safety
        variance = self.B - self.A**2
        
        # 4. Safe DSR Calculation
        # Add epsilon to denominator to prevent division by zero
        # Clip variance to be non-negative
        std_dev = np.sqrt(max(variance, 1e-6))
        
        dsr = self.A / std_dev
        
        # 5. Output Clipping
        # Prevent explosions like -20,000. Clamp to typical range [-5, 5]
        # This gives a strong signal but doesn't break the neural net
        dsr = np.clip(dsr, -5.0, 5.0)
        
        return dsr

    def _next_observation(self):
        # Get current market data for all stocks: shape (n_stocks, n_features)
        current_market_data = self.data[self.current_step]
        
        # Reshape shares_held to (n_stocks, 1) for horizontal stacking
        shares_reshaped = self.shares_held.reshape(-1, 1)
        
        # Combine market data with shares: shape (n_stocks, n_features + 1)
        # Each row is [Feature1, Feature2, ..., FeatureN, Shares]
        stock_obs = np.hstack((current_market_data, shares_reshaped))
        
        # Flatten and append balance
        obs = np.concatenate([stock_obs.flatten(), [self.balance]])
        
        return obs.astype(np.float32)