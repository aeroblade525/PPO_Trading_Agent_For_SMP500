import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SP500TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance=100000):
        super(SP500TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation Space: [Open, High, Low, Close, RSI, MACD, Balance, Shares]
        # We use float32 to match PPO expectations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # DSR Hyperparameters
        self.eta = 1 / 252  # Decay rate for DSR (approx 1 year window)
        
        # Initialize internal variables
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.A = 0.0
        self.B = 0.0
        self.prev_net_worth = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        
        # Reset DSR moving averages
        self.A = 0.0
        self.B = 0.0
        self.prev_net_worth = self.initial_balance
        
        obs = self._next_observation()
        info = {}
        return obs, info

    def step(self, action):
        # 1. Execute Action at CURRENT Price
        current_price = self.df.iloc[self.current_step]["Close"]
        self._take_action(action, current_price)
        
        # 2. Move to NEXT Day
        self.current_step += 1
        
        # 3. Check for Termination (End of Data)
        # We stop at len-1 because we need 'tomorrow's' price for reward calc
        if self.current_step >= len(self.df):
            return np.zeros(8, dtype=np.float32), 0, True, False, {"net_worth": self.prev_net_worth}

        # 4. Calculate Net Worth using NEW Price
        next_price = self.df.iloc[self.current_step]["Close"]
        current_val = self.balance + (self.shares_held * next_price)
        
        # 5. Bankruptcy Check
        bankrupt = current_val <= 0
        terminated = bankrupt

        # 6. Calculate Reward (Differential Sharpe Ratio)
        reward = self._calculate_dsr(current_val, self.prev_net_worth)
        
        # Bankruptcy penalty override
        if bankrupt:
            reward = -10.0
        
        # Update previous net worth for next step
        self.prev_net_worth = current_val

        # 7. Get New Observation
        obs = self._next_observation()
        info = {"net_worth": current_val}

        return obs, reward, terminated, False, info

    def _take_action(self, action, current_price):
        action_type = action
        
        # 1 = BUY
        if action_type == 1: 
            # Aggressive: Bet 90% of available cash
            # This fixes the "Cash Drag" where the agent only buys 1 share
            total_possible = self.balance
            shares_to_buy = total_possible // current_price
            
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
        
        # 2 = SELL
        elif action_type == 2: 
            # Sell Everything (Exit Position)
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        
        # 0 = HOLD (Do nothing)

    def _next_observation(self):
        # Safety check to prevent index crash
        if self.current_step >= len(self.df):
            return np.zeros(8, dtype=np.float32)

        frame = self.df.iloc[self.current_step]
        
        obs = np.array([
            frame['Open'], frame['High'], frame['Low'], frame['Close'],
            frame['RSI'], frame['MACD'],
            self.balance, self.shares_held
        ], dtype=np.float32)
        
        return obs

    def _calculate_dsr(self, current_val, prev_val):
        """Calculates Differential Sharpe Ratio"""
        if prev_val == 0: return 0
        
        # Log Returns for stability
        try:
            R_t = np.log(current_val / prev_val)
        except ValueError:
            return -1 # Handle negative/zero values gracefully

        delta_A = R_t - self.A
        delta_B = (R_t ** 2) - self.B
        
        # Store old values for gradient calculation
        prev_A = self.A
        prev_B = self.B
        
        # Update Moving Averages
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        
        # DSR Calculation
        variance = prev_B - (prev_A ** 2)
        
        # Stability: If variance is near zero, return 0 to avoid explosion
        if variance < 1e-6:
            return 0.0
            
        numerator = prev_B * delta_A - 0.5 * prev_A * delta_B
        denominator = variance ** 1.5
        
        D_t = numerator / denominator
        
        # Clip reward to keep PPO stable (-5 to +5)
        return np.clip(D_t, -5, 5)