# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Path to pre-generated market data
mock_data_path = "/Users/leo/PPO_Trading_Agent_For_SMP500/mock_market_data.npy"


# -------------------------------------------------
# Normalization wrapper for trading observations
# -------------------------------------------------
class SimpleNormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def normalize(self, obs):
        # Observation structure:
        # [Stock0_Data(7)+Shares(1), Stock1..., Balance(1)]
        # Total per stock = 8
        obs = np.array(obs, dtype=np.float32)

        n_features_plus_shares = 8
        n_stocks = (len(obs) - 1) // n_features_plus_shares

        # Normalize stock data
        for i in range(n_stocks):
            base = i * n_features_plus_shares

            # Prices: Open, High, Low, Close
            obs[base:base + 4] /= 1000.0

            # RSI
            obs[base + 4] /= 100.0

            # MACD
            obs[base + 5] /= 20.0

            # Volume
            obs[base + 6] /= 1_000_000.0

            # Shares held
            obs[base + 7] /= 100.0

        # Normalize balance
        obs[-1] /= 100_000.0
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info


# -------------------------------------------------
# Experiment arguments
# -------------------------------------------------
@dataclass
class Args:
    exp_name: str = "ppo_trading"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # Algorithm specific arguments
    total_timesteps: int = 237_500
    learning_rate: float = 3e-4
    num_envs: int = 5
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # Optional for tyro compatibility
    rpo_alpha: float = 0.5

    # Runtime computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# -------------------------------------------------
# Environment factory
# -------------------------------------------------
def make_env(gamma):
    def thunk():
        from src.env.environment import MultiStockTradingEnv

        data_tensor = np.load(mock_data_path)
        n_stocks = data_tensor.shape[1]

        # Create trading environment
        env = MultiStockTradingEnv(
            data_tensor=data_tensor,
            n_stocks=n_stocks,
            reward_type="dsr",
        )

        # Standard CleanRL wrappers
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
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))

        return env

    return thunk


# -------------------------------------------------
# Layer initialization
# -------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# -------------------------------------------------
# PPO Agent
# -------------------------------------------------
class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha):
        super().__init__()
        self.rpo_alpha = rpo_alpha

        obs_dim = np.prod(envs.single_observation_space.shape)
        act_dim = np.prod(envs.single_action_space.shape)

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor mean network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # Log standard deviation (learned)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
        else:
            # RPO noise injection
            z = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            dist = Normal(mean + z, std)

        return (
            action,
            dist.log_prob(action).sum(1),
            dist.entropy().sum(1),
            self.critic(x),
        )


# -------------------------------------------------
# Main training loop
# -------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)

    # Derived quantities
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    os.makedirs("models", exist_ok=True)

    # Startup sanity prints
    print("=" * 60)
    print("Starting PPO Trading Run")
    print(f"Run name: {run_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'}")
    print(f"Num envs: {args.num_envs}")
    print(f"Steps per rollout: {args.num_steps}")
    print(f"Total timesteps: {args.total_timesteps}")
    print("=" * 60)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Vectorized environments
    envs = gym.vector.SyncVectorEnv([make_env(args.gamma) for _ in range(args.num_envs)])

    agent = Agent(envs, args.rpo_alpha).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_scalar("debug/started", 1.0, 0)
    writer.flush()

    # Storage buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    best_episodic_return = -float("inf")

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    update_bar = trange(1, args.num_iterations + 1, desc="PPO Updates", unit="update")

    for update in update_bar:
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Environment step
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device)

            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.tensor(terminations | truncations).to(device)

            # Periodic reward print + logging
            if global_step % 5000 == 0:
                print(f"[Step {global_step}] Mean reward: {reward.mean():.4f}")
                writer.add_scalar("step/reward_mean", reward.mean(), global_step)

            # Log net worth if provided by env
            if "net_worth" in infos and global_step % 5000 == 0:
                mean_nw = float(np.mean(infos["net_worth"]))
                print(f"[Step {global_step}] Mean net worth: {mean_nw:.2f}")
                writer.add_scalar("portfolio/net_worth_mean", mean_nw, global_step)

            # Episode termination handling
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        net_worth = info.get("net_worth", None)

                        print(
                            f"[Episode End] Step={global_step} | "
                            f"Return={ep_return:.2f} | "
                            f"Len={ep_len} | "
                            f"NetWorth={net_worth}"
                        )

                        writer.add_scalar("charts/episodic_return", ep_return, global_step)

                        # Save best model
                        if ep_return > best_episodic_return:
                            best_episodic_return = ep_return
                            torch.save(
                                {
                                    "agent": agent.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "best_return": best_episodic_return,
                                },
                                f"models/{run_name}_best.pth",
                            )
                            print("✅ New best model saved")

        # Progress bar update
        update_bar.set_postfix(
            steps=global_step,
            sps=int(global_step / (time.time() - start_time)),
            best=f"{best_episodic_return:.2f}",
        )

    envs.close()
    writer.close()