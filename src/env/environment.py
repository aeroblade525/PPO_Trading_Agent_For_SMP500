import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_tensor, initial_balance=100000, n_stocks=100):
        super(MultiStockTradingEnv, self).__init__()
        
        # Data Structure: 
        # numpy array of shape (n_timesteps, n_stocks, n_features)
        # Features usually: [Open, High, Low, Close, RSI, MACD]
        self.data = data_tensor
        self.n_stocks = n_stocks
        self.n_features = data_tensor.shape[2]
        self.initial_balance = initial_balance
        
        # --- ACTION SPACE ---
        # A vector of 100 continuous numbers between -1 and 1
        # Positive = Buy weight, Negative = Sell percentage
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        # --- OBSERVATION SPACE ---
        # We see: (Market Data for 100 stocks) + (Shares Held for 100 stocks) + (Cash Balance)
        # Size = (100 * 6 features) + (100 share counts) + 1 balance
        obs_size = (self.n_stocks * self.n_features) + self.n_stocks + 1
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
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks, dtype=np.float32)
        self.current_step = 0
        self.prev_net_worth = self.initial_balance
        self.A = 0.0
        self.B = 0.0
        
        return self._next_observation(), {}

    def step(self, actions):
        # 1. Get Current Prices for all 100 stocks
        # Assuming 'Close' is at index 3 in our features
        current_prices = self.data[self.current_step, :, 3] 
        
        # 2. Execute Trades
        self._take_action(actions, current_prices)
        
        # 3. Move Time Forward
        self.current_step += 1
        
        # 4. Check Termination
        if self.current_step >= len(self.data) - 1:
            return np.zeros(self.observation_space.shape), 0, True, False, {"net_worth": self.prev_net_worth}

        # 5. Calculate Portfolio Value
        # Next Step Prices (to calculate immediate PnL)
        next_prices = self.data[self.current_step, :, 3]
        
        # Vectorized Value Calculation: Cash + (Vector of Shares * Vector of Prices)
        stock_value = np.sum(self.shares_held * next_prices)
        total_net_worth = self.balance + stock_value
        
        # 6. Reward (DSR on Total Portfolio)
        reward = self._calculate_dsr(total_net_worth, self.prev_net_worth)
        # Scale reward for stability
        reward *= 10.0 
        
        self.prev_net_worth = total_net_worth
        
        # Bankruptcy Check
        terminated = total_net_worth <= 0
        if terminated: reward = -10.0

        return self._next_observation(), reward, terminated, False, {"net_worth": total_net_worth}

    def _take_action(self, actions, current_prices):
        # actions is a vector of 100 floats (-1 to 1)
        
        # --- PHASE 1: SELLING ---
        # We process sells first to generate cash for buys
        sell_indices = np.where(actions < 0)[0]
        
        for idx in sell_indices:
            # action -0.5 means "Sell 50% of my position in this stock"
            sell_fraction = abs(actions[idx]) 
            shares_to_sell = self.shares_held[idx] * sell_fraction
            
            # Update state
            self.shares_held[idx] -= shares_to_sell
            self.balance += shares_to_sell * current_prices[idx]

        # --- PHASE 2: BUYING ---
        buy_indices = np.where(actions > 0)[0]
        
        if len(buy_indices) > 0:
            buy_weights = actions[buy_indices]
            
            # Softmax-style Normalization
            # If agent wants to buy Stock A (1.0) and Stock B (1.0), 
            # we can't spend 200% of cash. We split 50/50.
            total_weight = np.sum(buy_weights)
            
            # If total desire > 1, normalize to 1. 
            # If total desire is 0.5, we only spend 50% of cash (hold the rest).
            scale = 1.0
            if total_weight > 1.0:
                scale = 1.0 / total_weight
                
            # Execute Buys
            for idx, weight in zip(buy_indices, buy_weights):
                allocation_amt = self.balance * (weight * scale)
                
                # Transaction: floor division to get whole shares
                if current_prices[idx] > 0:
                    shares = allocation_amt // current_prices[idx]
                    
                    self.shares_held[idx] += shares
                    self.balance -= shares * current_prices[idx]

    def _next_observation(self):
        market_data = self.data[self.current_step].flatten()

        obs = np.concatenate([
            market_data,
            self.shares_held,
            [self.balance]
        ])

        return obs.astype(np.float32)