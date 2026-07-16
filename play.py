"""
play.py — Loads a trained DQN model and plays it in the Atari environment using
a greedy policy (deterministic=True, no exploration).

  --render-mode human   -> live GUI window
  --record              -> headless, saves an .mp4 (use on Colab)

Usage:
  python play.py --model-path models/dqn_model.zip --policy CnnPolicy --episodes 5
  python play.py --model-path models/dqn_model.zip --policy CnnPolicy --record --episodes 3
"""
import argparse
import os

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder

from envs import build_vec_env


def parse_args():
    p = argparse.ArgumentParser(description="Play an Atari environment with a trained DQN agent.")
    p.add_argument("--model-path", type=str, default="models/dqn_model.zip")
    p.add_argument("--env", type=str, default="Pong")
    p.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--record", action="store_true", help="Save an mp4 instead of live rendering")
    p.add_argument("--video-folder", type=str, default="videos")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def build_play_env(args):
    render_mode = "rgb_array" if args.record else args.render_mode
    env = build_vec_env(args.env, args.policy, seed=args.seed, n_envs=1, render_mode=render_mode)

    if args.record:
        os.makedirs(args.video_folder, exist_ok=True)
        # Generous cap - episodes can run long, and env.close() ends the file at the real finish anyway.
        video_length = args.episodes * 100_000
        env = VecVideoRecorder(
            env, args.video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=f"{args.env}_{args.policy}_play",
        )
    return env


def play(args) -> list:
    model = DQN.load(args.model_path)
    env = build_play_env(args)

    episode_rewards = []
    obs = env.reset()
    for ep in range(args.episodes):
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # GreedyQPolicy: argmax Q, no exploration
            obs, reward, done_arr, info = env.step(action)
            ep_reward += reward[0]
            done = bool(done_arr[0])
            if args.render_mode == "human" and not args.record:
                env.render()
        episode_rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{args.episodes}: reward = {ep_reward:.1f}")

    env.close()
    print(f"\nMean reward over {args.episodes} episodes: {np.mean(episode_rewards):.2f} "
          f"+/- {np.std(episode_rewards):.2f}")
    if args.record:
        print(f"Video saved to: {os.path.abspath(args.video_folder)}")
    return episode_rewards


if __name__ == "__main__":
    play(parse_args())
