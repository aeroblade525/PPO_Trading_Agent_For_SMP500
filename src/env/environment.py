import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SP500TradingEnv(gym.Env):
  metadata = {"render_modes":["human"]}

  def __init__(self, df, initial_balance=10000):
    super(SP500TradingEnv, self).__init__()

    self.df = df
    self.initial_balance = initial_balance
    
    self.action_space = spaces.Discrete(3)

    self.observation_space = spaces.Box(
      low=-np.inf, high = np.inf, shape = (8,), dtype=np.float32     #prob will change in future
    )


    self.eta = 1 / 252  # Decay rate (???) use 1/50 or 1/100 for faster adaptation (???)

    # DSR State Variables (Moving averages)
    self.A = 0.0
    self.B = 0.0

  def reset(self, seed = None, options = None):
    """
    Reset the environment to the starting state (day 0)
    """
    super().reset(seed = seed)
    
    self.balance = self.initial_balance
    self.shares_held = 0
    self.current_step = 0

    self.A = 0.0
    self.B = 0.0

    self.prev_net_worth = self.initial_balance

    obs = self._next_observation()
    info = {}

    return obs, info


  def _calculate_dsr(self, current_portfolio_val, prev_portfolio_val):
        """
        Calculates the Differential Sharpe Ratio (DSR) for the current step.
        """
        
        # 1. Calculate Returns (R_t)
        # Using logarithmic returns is preferred for mathematical stability
        # Add epsilon to avoid division by zero if values are identical
        if prev_portfolio_val == 0: return 0
        
        # R_t = Returns at time t
        R_t = np.log(current_portfolio_val / prev_portfolio_val)
        
        # 2. Update Moving Averages (Exponential Moving Average)
        # We need these from the *previous* step to calculate the gradient, 
        # but we update them *after* usually.
        # However, for the standard iterative formula (Eq 12 in the paper), 
        # we update the estimates based on the new R_t first.
        
        delta_A = R_t - self.A
        delta_B = (R_t ** 2) - self.B
        
        # Update A (First Moment) and B (Second Moment)
        prev_A = self.A  # Store old A for the calculation
        prev_B = self.B  # Store old B for the calculation
        
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B
        
        # 3. Calculate DSR (The Reward)
        # Formula derived from taking the derivative of Sharpe w.r.t the decay rate
        # D_t = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / (B_{t-1} - A_{t-1}^2)^(3/2)
        
        # Denominator: Variance^(3/2)
        variance = prev_B - (prev_A ** 2)
        
        # Stability check: If variance is tiny or negative (precision errors), return 0
        if variance < 1e-9:
            return 0.0
            
        denominator = variance ** 1.5
        
        # Numerator
        term1 = prev_B * delta_A
        term2 = 0.5 * prev_A * delta_B
        numerator = term1 - term2
        
        D_t = numerator / denominator
        
        return D_t

  def step(self, action):
    """
    Excecutes one time step
    """
    #1. Excecute the action (buy, sell, or hold) based on current price
    current_price = self.df.iloc[self.current_step]["Close"]
    self._take_action(action, current_price)

    #2. Move to the next step
    self.current_step += 1

    #3. Check if we are done (out of data/money)
    reached_end = self.current_step >= len(self.df) - 1
    
    current_val = self.balance + (self.shares_held * current_price)
    bankrupt = current_val <= 0
    terminated = reached_end or bankrupt

    #4. Calculate reward (prob will need to tweak)
    reward = self._calculate_dsr(current_val, self.prev_net_worth)
    if bankrupt:
      reward = -10

    self.prev_net_worth = current_val

    #5. Get new observation
    obs = self._next_observation()
    info = {"net_worth": current_val}

    return obs, reward, terminated, False, info

  def _next_observation(self):
    """
    Helper to return what agent sees for the current step
    """

    # Get the row of data
    frame = self.df.iloc[self.current_step]

    # Contruct the array matching self.observation_space 
    obs = np.array([
            frame['Open'],
            frame['High'],
            frame['Low'],
            frame['Close'],
            frame['RSI'],
            frame['MACD'],
            self.balance,
            self.shares_held
        ], dtype=np.float32)

    return obs

  def _take_action(self, action, current_price):
        # Action Set: 0 = Hold, 1 = Buy, 2 = Sell
        
        action_type = action
        
        # Tuning parameter: How much of our balance do we bet per trade?
        # A simple agent bets a fixed percentage (e.g., 10%) or a fixed amount.
        # Let's say we buy/sell 1 share at a time for simplicity, 
        # or we can calculate max shares manageable.
        if action_type == 1: # Buy
            # Calculate max shares we can afford
            # We add a small buffer so we don't go to exactly 0 (floating point errors)
            max_shares = self.balance // current_price
            
            # For this example, let's try to buy 1 share if we can afford it
            if max_shares > 0:
                self.balance -= current_price
                self.shares_held += 1
        
        elif action_type == 2: # Sell
            # We can only sell if we have shares
            if self.shares_held > 0:
                self.balance += current_price
                self.shares_held -= 1
        
        # If action is 0 (Hold), we do nothing.