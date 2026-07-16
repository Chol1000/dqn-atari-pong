# Formative 3 — Deep Q-Learning on Atari (Pong)

DQN agent trained with Stable Baselines3 on Pong, comparing `CnnPolicy` (pixel frames) vs `MlpPolicy` (RAM state), with a 30-experiment hyperparameter sweep across the team.

**Team:** Chol (learning rate & batch size), Lorita (gamma & epsilon-greedy schedule), Josephine (replay buffer/learning-starts & CNN vs MLP). This initial push covers Chol's own contribution — code, infrastructure, the final model, and Chol's 10 experiments. Lorita's and Josephine's experiments will be added via their own commits.

## Setup

```bash
pip install -r requirements.txt
```
GPU recommended (trained on Colab Pro, L4 GPU). Full pipeline: `notebooks/Formative_3_DQN_Colab.ipynb`.

## Usage

```bash
# Train
python train.py --member "Chol" --exp-id 1 --policy CnnPolicy \
  --lr 1e-4 --gamma 0.99 --batch-size 32 \
  --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 0.1 --total-timesteps 300000

# Play (greedy policy: deterministic=True, no exploration)
python play.py --model-path models/dqn_model.zip --policy CnnPolicy --episodes 5
```

## Gameplay video
<!-- VIDEO_START -->
[Pong_CnnPolicy_play-step-0-to-step-300000.mp4](videos/Pong_CnnPolicy_play-step-0-to-step-300000.mp4)
<!-- VIDEO_END -->

## Hyperparameter Tuning Results — Chol (learning rate & batch size)
<!-- TABLE_START -->
| Member | Exp # | Policy | lr | gamma | batch_size | epsilon_start | epsilon_end | epsilon_decay | Mean Reward | Mean Ep. Length | Noted Behavior |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chol | 1 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | Baseline configuration (SB3 default-like values) |
| Chol | 2 | CnnPolicy | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | Higher lr may speed up early learning but risks instability |
| Chol | 3 | CnnPolicy | 1e-05 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | Lower lr should be more stable but slower to converge |
| Chol | 4 | CnnPolicy | 0.0001 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | -20.8 | 817.2 | Smaller batch -> noisier gradient estimates |
| Chol | 5 | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | -10.8 | 2219.6 | Larger batch -> smoother gradients and more compute per step |
| Chol | 6 | CnnPolicy | 0.0001 | 0.99 | 128 | 1.0 | 0.01 | 0.1 | -9.8 | 2910.8 | Even larger batch: expect diminishing returns vs compute cost |
| Chol | 7 | CnnPolicy | 0.001 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | Combo: high lr + large batch to offset gradient noise |
| Chol | 8 | CnnPolicy | 1e-05 | 0.99 | 16 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | Combo: low lr + small batch - expect slow/underfit learning |
| Chol | 9 | CnnPolicy | 0.00025 | 0.99 | 32 | 1.0 | 0.01 | 0.1 | -21.0 | 757.2 | SB3-Zoo-style lr for comparison against our baseline |
| Chol | 10 | CnnPolicy | 0.0001 | 0.99 | 256 | 1.0 | 0.01 | 0.1 | 16.0 | 1872.8 | Very large batch: stability vs wall-clock time trade-off |
<!-- TABLE_END -->

_Lorita's and Josephine's experiment rows (20 more) will be added here via their own commits._

### Discussion
<!-- DISCUSSION_START -->
**Batch size was the single biggest factor** — a monotonic dose-response: 16 → -20.8, 32 → -21.0, 64 → -10.8, 128 → -9.8, 256 → **+16.0** (the best result of the whole sweep). Learning rate alone (0.001 or 1e-5) made no difference at batch=32, so this wasn't just "more tuning helps" — larger batches give a smoother, less noisy gradient estimate per update, which is why learning actually compounded instead of getting knocked around at every step. Combining a larger batch with a higher learning rate (exp7) cancelled the benefit out, and the reverse combo (exp8) stayed flat too — batch size only helps paired with a stable learning rate, not as an independent lever. Best final config: `lr=0.0001, gamma=0.99, batch_size=256`. Full write-up: [`docs/analysis_chol.md`](docs/analysis_chol.md).

**Final model:** `batch_size=256` carried forward to a 1,000,000-step production run, finishing at `mean_reward = 10.40 +/- 7.26` — confirming the effect held at ten times the training budget, not just as a lucky result at 300k steps.
<!-- DISCUSSION_END -->
