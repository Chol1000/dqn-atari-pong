# Josephine's Analysis — Replay Buffer, Learning-Starts & CNN vs MLP Architecture

I ran 10 experiments covering two different questions: how replay buffer size and learning-starts timing affect training, and how CnnPolicy (pixels) compares to MlpPolicy (RAM) as an architecture choice.

## What I tested
- Buffer size: 10,000 (smaller), 150,000 (larger) (baseline 100,000)
- Learning-starts: 1,000 (earlier), 25,000 (later) (baseline 10,000)
- Architecture: MlpPolicy on RAM vs CnnPolicy on pixels, at matching hyperparameters
- Combinations: large buffer + large batch, MlpPolicy + reduced gamma
## Architecture: the clearest result in my entire block
Every single MlpPolicy (RAM) experiment I ran — exp2, 7, 8, and 10 — flatlined at exactly -21.0 reward, with a near-identical, very short episode length (~757-792 steps), regardless of whether I changed the learning rate (0.001 vs 0.0001) or batch size (32 vs 64). The RAM-based MLP simply never learned to survive a rally within this training budget. CnnPolicy runs, by contrast, varied widely across the whole sweep and several broke past -15. This is an unambiguous answer to the assignment's architecture-comparison question: CnnPolicy on pixels was clearly the stronger choice for this environment.

## Buffer size and learning-starts: the opposite of what I predicted
My hypotheses going in were that a smaller buffer would hurt (too correlated/repetitive samples) and starting learning later would waste time. The actual results ran the other way:

| Change | mean_reward | vs. baseline (-21.0) |
|---|---|---|
| Smaller buffer (10,000) | -16.4 | better |
| Larger buffer (150,000) | -19.2 | slightly better |
| Learning-starts early (1,000) | -15.0 | better |
| Learning-starts late (25,000) | -14.4 | better (best in my block) |
| Combo: large buffer + large batch | -14.0 | better (second-best) |
Both a smaller buffer and a later learning-starts point outperformed the baseline, which is the opposite of what I predicted. I think what's actually happening is that delaying when gradient updates begin gives the replay buffer more time to fill with diverse experience before the network starts learning from it, avoiding early overfitting to a small, low-diversity buffer — the same mechanism likely explains why the smaller buffer wasn't as harmful as expected, since it wasn't the buffer's *size* that mattered as much as how early and how repetitive the data being sampled was.

## Final takeaway
The two most useful things I found: architecture choice mattered enormously (CNN vs MLP), and my assumptions about buffer/learning-starts were wrong in an informative way — a real result worth reporting honestly rather than a shortfall.
