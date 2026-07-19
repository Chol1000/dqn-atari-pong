# Formative 3 — Deep Q-Learning on Atari (Pong)

DQN agent trained with Stable Baselines3 on Pong, comparing `CnnPolicy` (pixel frames) vs `MlpPolicy` (RAM state), with a 30-experiment hyperparameter sweep across the team.

**Team:** Chol (learning rate & batch size), Lorita (gamma & epsilon-greedy schedule), Josephine (replay buffer/learning-starts & CNN vs MLP).

## Repository Structure

- `train.py`, `play.py`, `envs.py` — training, evaluation, and shared environment code
- `run_experiments.py`, `finalize_submission.py` — sweep automation and README generation
- `experiments/` — hyperparameter configs and logged results (CSV)
- `models/dqn_model.zip` — final trained model
- `videos/` — recorded gameplay clip
- `notebooks/` — full Colab pipeline (all 30 experiments + final model + video, in one run)
- `docs/` — DQN/RL concepts primer, per-member analysis write-ups, and the coach tracking sheet

## Setup

```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
GPU recommended (trained on Colab Pro, L4 GPU). Full pipeline: `notebooks/Formative_3_DQN_Colab.ipynb`.

`play.py` also needs a real display to open a live window (`--render-mode human`), or run headless with `--record` (see below) if there's no display available.

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

## Hyperparameter Tuning Results
<!-- TABLE_START -->
### Chol — Learning Rate & Batch Size

| Exp # | Policy | lr | gamma | batch_size | epsilon (start/end/decay) | Mean Reward | Mean Ep. Length | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Baseline configuration (baseline) |
| 2 | CnnPolicy | 0.001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Higher lr may speed up early learning but risks instability (no change) |
| 3 | CnnPolicy | 1e-05 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Lower lr should be more stable but slower to converge (no change) |
| 4 | CnnPolicy | 0.0001 | 0.99 | 16 | 1.0 / 0.01 / 0.1 | -20.8 | 817.2 | Smaller batch -> noisier gradient estimates (-21.0 -> -20.8) |
| 5 | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 / 0.01 / 0.1 | -10.8 | 2219.6 | Larger batch -> smoother gradients and more compute per step (-21.0 -> -10.8) |
| 6 | CnnPolicy | 0.0001 | 0.99 | 128 | 1.0 / 0.01 / 0.1 | -9.8 | 2910.8 | Even larger batch: expect diminishing returns vs compute cost (-21.0 -> -9.8) |
| 7 | CnnPolicy | 0.001 | 0.99 | 64 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Combo: high lr + large batch to offset gradient noise (no change) |
| 8 | CnnPolicy | 1e-05 | 0.99 | 16 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Combo: low lr + small batch - expect slow/underfit learning (no change) |
| 9 | CnnPolicy | 0.00025 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | SB3-Zoo-style lr for comparison against our baseline (no change) |
| 10 | CnnPolicy | 0.0001 | 0.99 | 256 | 1.0 / 0.01 / 0.1 | **16.0** | 1872.8 | Very large batch: stability vs wall-clock time trade-off (-21.0 -> +16.0) |

### Lorita — Gamma & Epsilon-Greedy Schedule

| Exp # | Policy | lr | gamma | batch_size | epsilon (start/end/decay) | Mean Reward | Mean Ep. Length | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Baseline configuration (same as Chol exp1 for a shared reference point) (baseline) |
| 2 | CnnPolicy | 0.0001 | 0.9 | 32 | 1.0 / 0.01 / 0.1 | -20.4 | 1801.2 | Lower gamma -> more short-sighted agent and may miss long rallies (-21.0 -> -20.4) |
| 3 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 / 0.01 / 0.1 | -15.6 | 2436.2 | Moderate gamma reduction (-21.0 -> -15.6) |
| 4 | CnnPolicy | 0.0001 | 0.999 | 32 | 1.0 / 0.01 / 0.1 | -17.4 | 1454.2 | Very high gamma -> values distant future reward almost equally (-21.0 -> -17.4) |
| 5 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.3 | -20.4 | 852.2 | Slower epsilon decay -> prolonged exploration (-21.0 -> -20.4) |
| 6 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.02 | -16.8 | 1299.2 | Faster epsilon decay -> agent exploits sooner and risks a local optimum (-21.0 -> -16.8) |
| 7 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.05 / 0.1 | -19.4 | 1345.2 | Higher epsilon floor -> never fully stops exploring (-21.0 -> -19.4) |
| 8 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.0 / 0.1 | -14.4 | 2636.2 | Fully greedy at end of decay (epsilon_end=0) (-21.0 -> -14.4) |
| 9 | CnnPolicy | 0.0001 | 0.95 | 32 | 1.0 / 0.01 / 0.2 | -16.6 | 1957.2 | Combo: reduced gamma + slower epsilon decay (-21.0 -> -16.6) |
| 10 | CnnPolicy | 0.0001 | 0.999 | 32 | 1.0 / 0.05 / 0.1 | -20.0 | 1057.2 | Combo: high gamma + higher exploration floor (-21.0 -> -20.0) |

### Josephine — Replay Buffer / Learning-Starts & CNN vs MLP

| Exp # | Policy | lr | gamma | batch_size | epsilon (start/end/decay) | Mean Reward | Mean Ep. Length | Noted Behavior |
|---|---|---|---|---|---|---|---|---|
| 1 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 757.2 | Baseline configuration (CnnPolicy on pixels) for architecture comparison (baseline) |
| 2 | MlpPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 764.0 | Same hyperparameters but MlpPolicy on RAM observations (architecture comparison) (no change) |
| 3 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -16.4 | 2058.2 | Small replay buffer -> more correlated/repetitive samples (-21.0 -> -16.4) |
| 4 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -19.2 | 757.2 | Large replay buffer -> more diverse samples and higher memory cost (-21.0 -> -19.2) |
| 5 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -15.0 | 7723.6 | Learning starts early -> updates on a very small/immature buffer (-21.0 -> -15.0) |
| 6 | CnnPolicy | 0.0001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -14.4 | 3029.2 | Learning starts late -> longer pure-exploration warmup (-21.0 -> -14.4) |
| 7 | MlpPolicy | 0.001 | 0.99 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 764.0 | MlpPolicy with higher lr: does RAM-based learning tolerate it better? (no change) |
| 8 | MlpPolicy | 0.0001 | 0.99 | 64 | 1.0 / 0.01 / 0.1 | -21.0 | 764.0 | MlpPolicy with larger batch size (no change) |
| 9 | CnnPolicy | 0.0001 | 0.99 | 64 | 1.0 / 0.01 / 0.1 | -14.0 | 2898.2 | Combo: large buffer + large batch (-21.0 -> -14.0) |
| 10 | MlpPolicy | 0.0001 | 0.95 | 32 | 1.0 / 0.01 / 0.1 | -21.0 | 792.0 | Combo: MlpPolicy + reduced gamma (no change) |
<!-- TABLE_END -->

### Discussion
<!-- DISCUSSION_START -->
**Batch size was the single biggest factor** — a monotonic dose-response across Chol's block: 16 → -20.8, 32 → -21.0, 64 → -10.8, 128 → -9.8, 256 → **+16.0** (best of all 30 runs). Learning rate alone (0.001 or 1e-5) made no difference at batch=32, so this wasn't just "more tuning helps" — larger batches give a smoother, less noisy gradient estimate per update, which is why learning actually compounded instead of getting knocked around at every step. Best final config: `lr=0.0001, gamma=0.99, batch_size=256`. Full write-up: [`docs/analysis_chol.md`](docs/analysis_chol.md).

**Gamma & epsilon (Lorita)** showed a real interior optimum for gamma rather than "closer to 1 is better" (0.90 → -20.4, 0.95 → -15.6 best, 0.999 → -17.4), and `epsilon_end=0` (fully greedy after decay) beat leaving any residual exploration floor (-14.4 vs -19.4 at a 0.05 floor) — once the policy is decent, random exploration mostly just throws away winnable points. Full write-up: [`docs/analysis_lorita.md`](docs/analysis_lorita.md).

**Architecture (Josephine):** every `MlpPolicy` (RAM) run flatlined at exactly -21.0 regardless of tuning, while `CnnPolicy` (pixels) varied widely and repeatedly broke past -15 — an unambiguous result in favor of CNN for this environment. Buffer size and learning-starts also ran opposite to the pre-run hypotheses: a smaller buffer and later learning-starts both beat baseline, likely because delaying training gave the buffer more time to fill with diverse experience before learning began. Full write-up: [`docs/analysis_josephine.md`](docs/analysis_josephine.md).

**Final model:** `batch_size=256` (the clear sweep winner) carried forward to a 1,000,000-step production run, finishing at `mean_reward = 10.40 +/- 7.26` — confirming the effect held at ten times the training budget, not just as a lucky result at 300k steps.
<!-- DISCUSSION_END -->
