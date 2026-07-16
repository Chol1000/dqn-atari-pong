"""
train.py — Trains a DQN agent (Stable Baselines3) on an Atari environment and
appends one row to experiments/hyperparameter_results.csv per run.

  --policy CnnPolicy -> pixel frames (84x84 grayscale, 4-frame stack)
  --policy MlpPolicy -> 128-byte RAM state, normalized to [0, 1]

Usage:
  python train.py --member "Alice" --exp-id 1 --policy CnnPolicy \
      --lr 1e-4 --gamma 0.99 --batch-size 32 \
      --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 0.1 \
      --total-timesteps 300000
"""
import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from envs import build_eval_env, build_vec_env, pixel_env_id, ram_env_label

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "experiments", "hyperparameter_results.csv")
RESULTS_HEADER = [
    "timestamp", "member", "exp_id", "env_id", "policy",
    "lr", "gamma", "batch_size", "epsilon_start", "epsilon_end", "epsilon_decay",
    "buffer_size", "learning_starts", "total_timesteps",
    "mean_reward", "std_reward", "mean_ep_length", "train_seconds",
    "model_path", "noted_behavior",
]


def train(args) -> dict:
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    train_env = build_vec_env(args.env, args.policy, args.seed, args.n_envs)
    eval_env = build_eval_env(args.env, args.policy, args.seed)

    model = DQN(
        policy=args.policy,
        env=train_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )
    tb_log_name = f"{args.member}_exp{args.exp_id}_{args.policy}".replace(" ", "_")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.tensorboard_log, "best_model"),
        log_path=args.tensorboard_log,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = [eval_callback]
    if args.checkpoint_freq > 0:
        checkpoint_dir = os.path.join(os.path.dirname(args.model_path) or ".", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        name_prefix = f"{args.member}_exp{args.exp_id}".replace(" ", "_")
        callbacks.append(CheckpointCallback(save_freq=args.checkpoint_freq, save_path=checkpoint_dir, name_prefix=name_prefix))

    start = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=tb_log_name,
        progress_bar=args.progress_bar,
    )
    train_seconds = time.time() - start

    model.save(args.model_path)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    mean_ep_length = _mean_episode_length(eval_env, model, args.eval_episodes)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "member": args.member,
        "exp_id": args.exp_id,
        "env_id": pixel_env_id(args.env) if args.policy == "CnnPolicy" else ram_env_label(args.env),
        "policy": args.policy,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "total_timesteps": args.total_timesteps,
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "mean_ep_length": round(mean_ep_length, 1),
        "train_seconds": round(train_seconds, 1),
        "model_path": args.model_path,
        "noted_behavior": args.notes,
    }
    _append_result(result)
    print(f"\n[DONE] {args.member} exp{args.exp_id}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return result


def _mean_episode_length(eval_env, model, n_episodes):
    lengths = []
    obs = eval_env.reset()
    for _ in range(n_episodes):
        length = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done_arr, _ = eval_env.step(action)
            done = bool(done_arr[0])
            length += 1
        lengths.append(length)
    return float(np.mean(lengths))


def _append_result(result: dict):
    file_exists = os.path.isfile(RESULTS_CSV)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN agent on an Atari environment.")
    p.add_argument("--env", type=str, default="Pong", help="Base Atari env name, e.g. Pong")
    p.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    p.add_argument("--member", type=str, default="Unassigned", help="Group member running this experiment")
    p.add_argument("--exp-id", type=int, default=1, help="Experiment number (1-10) for this member")
    p.add_argument("--notes", type=str, default="", help="Qualitative observation for the hyperparameter table")

    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epsilon-start", type=float, default=1.0, dest="epsilon_start")
    p.add_argument("--epsilon-end", type=float, default=0.01, dest="epsilon_end")
    p.add_argument("--epsilon-decay", type=float, default=0.1, dest="epsilon_decay",
                    help="Fraction of total_timesteps over which epsilon decays start->end (SB3 exploration_fraction)")

    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=1_000)
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--checkpoint-freq", type=int, default=0,
                    help="Save a resumable checkpoint every N steps (0 = disabled). "
                         "Recommended for long runs (e.g. the final production model).")

    p.add_argument("--model-path", type=str, default="models/dqn_model.zip")
    p.add_argument("--tensorboard-log", type=str, default="tensorboard_logs")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
