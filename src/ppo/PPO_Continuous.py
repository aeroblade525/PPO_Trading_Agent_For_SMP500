# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
import yfinance as yf
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
mock_data_path = "/Users/leo/PPO_Trading_Agent_For_SMP500/mock_market_data.npy"

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
            obs[base:base + 4] /= 1000.0

            # RSI (Index 4)
            obs[base + 4] /= 100.0

            # MACD (Index 5)
            obs[base + 5] /= 20.0

            # Volume (Index 6) - Normalize by something large, e.g., 10M or 100M
            obs[base + 6] /= 1_000_000.0

            # Shares (Index 7)
            obs[base + 7] /= 100.0

        # Normalize Balance (Last element)
        obs[-1] /= 100000.0

        return obs
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info

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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "PPO"
    # """the id of the environment"""
    total_timesteps: int = 237_500
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        from src.env.environment import MultiStockTradingEnv
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
            data_tensor = np.expand_dims(df.values, axis=1)  # (Time, 1, 7)
            data_tensor = np.tile(data_tensor, (1, 100, 1))  # (Time, 100, 7)

            data_tensor = np.tile(data_tensor, (1, 100, 1))  # (Time, 100, 7)

            # Switch to DSR for advanced optimization
            raw_env = MultiStockTradingEnv(data_tensor=data_tensor, n_stocks=100, reward_type="dsr")
        env = raw_env
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        # initlalize critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # initalize actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # logarithm of the policy’s standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """gets the critics estimate of the reward for the action"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    # grabs params
    args = tyro.cli(Args)
    # sets batch size
    args.batch_size = int(args.num_envs * args.num_steps)
    # sets mini batch size
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # sets number of iterations
    args.num_iterations = args.total_timesteps // args.batch_size
    # nice start output text
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_scalar("debug/test", 1.0, 0)
    writer.flush()
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    best_episodic_return = -float("inf")
    os.makedirs("models", exist_ok=True)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    update_bar = trange(
        1,
        num_updates + 1,
        desc="PPO Updates",
        unit="update",
    )

    for update in update_bar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]

                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        # === BEST MODEL CHECKPOINT ===
                        if episodic_return > best_episodic_return:
                            best_episodic_return = episodic_return

                            torch.save(
                                {
                                    "agent": agent.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "global_step": global_step,
                                    "best_episodic_return": best_episodic_return,
                                },
                                f"models/{run_name}_best.pth",
                            )

                            print(
                                f"✅ New best model saved | "
                                f"Return: {episodic_return:.2f} | "
                                f"Step: {global_step}"
                            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))

        update_bar.set_postfix(
            steps=global_step,
            sps=sps,
            v_loss=f"{v_loss.item():.3f}",
            p_loss=f"{pg_loss.item():.3f}",
            kl=f"{approx_kl.item():.4f}",
        )
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/best_episodic_return", best_episodic_return, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()