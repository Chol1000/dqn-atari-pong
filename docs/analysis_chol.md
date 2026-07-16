# Chol's Analysis — Learning Rate & Batch Size

I ran 10 experiments varying learning rate and batch size, keeping everything else at the SB3-default-like baseline (gamma=0.99, epsilon 1.0→0.01 over 10% of training).

## What I tested
- Learning rate alone: 0.001, 1e-05, 0.00025 (baseline 0.0001)
- Batch size alone: 16, 64, 128, 256 (baseline 32)
- Combinations: high lr + large batch, low lr + small batch

## What actually happened
Learning rate on its own made no difference at all — every lr-only change (0.001, 1e-05, 0.00025) stayed pinned at exactly -21.0, identical to the baseline. Batch size, on the other hand, showed a clear, monotonic dose-response:

| batch_size | mean_reward |
|---|---|
| 16 | -20.8 |
| 32 (baseline) | -21.0 |
| 64 | -10.8 |
| 128 | -9.8 |
| 256 | **+16.0** |

256 was the best result across all 30 experiments in the entire sweep — more than double the next-best configuration from any member.

## The interesting exception
Combining a larger batch (64) with a higher learning rate (exp7) *cancelled out* the batch-size benefit entirely, landing back at -21.0. The reverse combo — low lr (1e-05) with a small batch (16, exp8) — also stayed at -21.0, confirming that neither weak lever alone nor combining two weak levers gets anywhere close to what the winning batch_size=256 does by itself. This tells me batch size only helps when paired with a stable learning rate — it's not simply "bigger batch = better" as an independent lever, the two interact.

exp9 (SB3-Zoo-style lr=0.00025, everything else baseline) was included specifically as a sanity check against a published, known-good DQN configuration for Atari — it also stayed at -21.0, reinforcing that at batch_size=32, no learning-rate choice by itself moves the result.

## Why I think this happened
Larger batches give a smoother, less noisy gradient estimate at each update — with only 32 samples, the gradient direction is noisy and the network struggles to make consistent progress. At 256, updates are far more stable, so learning actually compounds instead of getting knocked around. The trade-off is more compute per update, but at this training budget it was clearly worth it.

## Final configuration used
`lr=0.0001, gamma=0.99, batch_size=256` — this became the config for the team's final 1,000,000-step production model, which finished at `mean_reward = 10.40 +/- 7.26`, confirming the effect held up at scale, not just as a lucky result at 300k steps.
