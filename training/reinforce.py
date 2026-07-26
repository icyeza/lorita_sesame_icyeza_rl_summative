"""From-scratch PyTorch REINFORCE (Monte-Carlo policy gradient).

Stable-Baselines3 has no REINFORCE implementation, so this is hand-rolled:
full-episode Monte Carlo returns, a single policy network, and an optional
running-mean return baseline (off by default -- vanilla REINFORCE) behind
`use_baseline`. It exposes `.learn(total_timesteps)` / `.predict(obs)` /
`.save(path)` / `REINFORCE.load(path, env)` so it's a drop-in alongside the
Stable-Baselines3 algorithms in `training/pg_training.py` and
`evaluation/evaluate.py` -- one harness, four algorithms.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class REINFORCE:
    """Minimal from-scratch REINFORCE compatible with the DQN/A2C/PPO harness."""

    def __init__(self, env, lr: float = 3e-4, gamma: float = 0.99,
                 hidden: tuple[int, ...] = (64, 64), use_baseline: bool = False,
                 entropy_coef: float = 0.0, seed: int | None = None,
                 device: str = "cpu", tensorboard_log: str | None = None):
        self.env = env
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        if seed is not None:
            torch.manual_seed(seed)
        self.policy = PolicyNet(obs_dim, n_actions, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._baseline = 0.0
        self._baseline_momentum = 0.95
        self.tensorboard_log = tensorboard_log
        self.num_timesteps = 0
        self._episode_log: list[dict] = []

    def predict(self, obs, deterministic: bool = False, state=None, episode_start=None):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_t)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        action = action.cpu().numpy()
        return (int(action[0]) if single else action), None

    def _run_episode(self):
        obs, _ = self.env.reset()
        log_probs, rewards, entropies = [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            rewards.append(reward)
            done = terminated or truncated
        return log_probs, rewards, entropies

    def _returns(self, rewards: list[float]) -> torch.Tensor:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return torch.as_tensor(returns, device=self.device)

    def learn(self, total_timesteps: int, log_path: str | None = None, callback=None,
              max_wall_clock_seconds: float | None = None):
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            f = open(log_path, "w", newline="")
            writer = csv.writer(f)
            writer.writerow(["episode", "timesteps", "episode_reward", "episode_length", "entropy"])
        else:
            f, writer = None, None

        writer_tb = None
        if self.tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            writer_tb = SummaryWriter(self.tensorboard_log)

        episode = 0
        start_time = time.monotonic()
        while self.num_timesteps < total_timesteps:
            if max_wall_clock_seconds is not None and (time.monotonic() - start_time) >= max_wall_clock_seconds:
                break
            log_probs, rewards, entropies = self._run_episode()
            returns = self._returns(rewards)

            if self.use_baseline:
                self._baseline = (self._baseline_momentum * self._baseline
                                   + (1 - self._baseline_momentum) * returns.mean().item())
                advantage = returns - self._baseline
            else:
                advantage = returns

            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies).mean()
            loss = -(log_probs_t * advantage.detach()).sum() - self.entropy_coef * entropy_t

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.num_timesteps += len(rewards)
            episode += 1
            ep_reward = float(sum(rewards))
            self._episode_log.append(dict(episode=episode, timesteps=self.num_timesteps,
                                           episode_reward=ep_reward, episode_length=len(rewards),
                                           entropy=float(entropy_t.item())))
            if writer:
                writer.writerow([episode, self.num_timesteps, ep_reward, len(rewards), float(entropy_t.item())])
                f.flush()
            if writer_tb:
                writer_tb.add_scalar("rollout/ep_rew_mean", ep_reward, self.num_timesteps)
                writer_tb.add_scalar("train/entropy", float(entropy_t.item()), self.num_timesteps)
                writer_tb.add_scalar("train/loss", float(loss.item()), self.num_timesteps)
            if callback is not None:
                callback(self)

        if f:
            f.close()
        if writer_tb:
            writer_tb.close()
        return self

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            "state_dict": self.policy.state_dict(),
            "obs_dim": self.env.observation_space.shape[0],
            "n_actions": self.env.action_space.n,
        }, path if path.endswith(".pt") else path + ".pt")

    @classmethod
    def load(cls, path: str, env, device: str = "cpu"):
        path = path if path.endswith(".pt") else path + ".pt"
        ckpt = torch.load(path, map_location=device)
        model = cls(env, device=device)
        model.policy.load_state_dict(ckpt["state_dict"])
        return model
