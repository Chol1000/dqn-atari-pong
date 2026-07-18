# DQN & Reinforcement Learning Concepts

Background on the reinforcement learning and DQN concepts used in this project.

## 1. Reinforcement learning basics

- **Agent**: the DQN policy (a CNN or MLP predicting Q-values).
- **Environment**: the Atari emulator (`PongNoFrameskip-v4`), exposed through Gymnasium/ALE.
- **State**: 4 stacked grayscale 84x84 frames (CnnPolicy) or the 128-byte console RAM (MlpPolicy).
- **Action**: one of 6 discrete joystick moves (NOOP, FIRE, UP, DOWN, ...).
- **Reward**: +1 when the agent scores, -1 when the opponent scores, 0 otherwise.
- **Episode**: one Pong match, ends when a side reaches 21 points (or `truncated` on time-limit).

## 2. Q-learning -> Deep Q-learning

Q-learning learns an action-value function `Q(s, a)`: the expected discounted future reward of taking action `a` in state `s` and then acting optimally. The update rule (Bellman equation) is:

```
Q(s, a) <- Q(s, a) + lr * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

Classic Q-learning stores `Q` in a table — impossible here because the state space (raw pixels) is effectively infinite. **Deep** Q-learning replaces the table with a neural network `Q(s, a; theta)` trained to minimize the TD (temporal-difference) error:

```
loss = ( r + gamma * max_a' Q(s', a'; theta_target) - Q(s, a; theta) )^2
```

This is exactly what `stable_baselines3.DQN` implements.

## 3. Why two networks (online + target)?

If the same network produced both the prediction `Q(s,a;theta)` and the target `r + gamma*max Q(s',a';theta)`, the target would shift every gradient step ("chasing a moving target"), destabilizing training. SB3 keeps a **target network**, copied from the online network every `target_update_interval` steps, so the regression target stays fixed for a while. This is why `target_update_interval` is a hyperparameter worth tuning: too frequent -> instability; too infrequent -> slow propagation of new information.

## 4. Why a replay buffer?

Consecutive frames of gameplay are highly correlated (the ball barely moves between frames), which violates the i.i.d. assumption SGD relies on and can cause the network to overfit to whatever it's currently seeing. The **replay buffer** stores past transitions `(s, a, r, s', done)` and samples random minibatches for training, which:
1. Breaks temporal correlation between consecutive updates.
2. Reuses each transition for multiple updates (data efficiency), since real environment interaction is expensive.

`buffer_size` controls how much history is kept; `learning_starts` controls how many random-policy transitions are collected before training begins (a small warm-up so the buffer isn't empty/tiny at the start).

## 5. Exploration vs. exploitation — epsilon-greedy

At every step the agent picks a random action with probability `epsilon`, otherwise the greedy action `argmax_a Q(s,a)`. Early in training `epsilon` is high (explore broadly, since the Q-estimates are meaningless); it decays over training so the agent increasingly exploits what it has learned.

SB3 parameterizes this with three knobs, mapped to the assignment's terms:

| Assignment term | SB3 parameter | Meaning |
|---|---|---|
| `epsilon_start` | `exploration_initial_eps` | epsilon at step 0 |
| `epsilon_end` | `exploration_final_eps` | epsilon floor after decay finishes |
| `epsilon_decay` | `exploration_fraction` | fraction of `total_timesteps` over which epsilon linearly decays from start to end |

Trade-off: decaying too fast risks locking in a suboptimal policy before the agent has seen enough of the state space (premature exploitation); decaying too slow wastes training time on random play once the agent already has a good policy.

**At evaluation time (`play.py`) there is no exploration at all** — `model.predict(obs, deterministic=True)` always takes `argmax_a Q(s,a)`. This *is* the "GreedyQPolicy" the assignment refers to; SB3 doesn't need a separate class for it because greedy action selection is just `deterministic=True` with epsilon effectively 0.

**Exact schedule mechanics (verified against the SB3 source, not assumed):** before `learning_starts` (default 10,000 steps), actions are fully uniform-random regardless of epsilon — there's no Q-network worth exploiting yet. `exploration_fraction` counts from step 0 of `total_timesteps`, not from `learning_starts`. So with the sweep's defaults (300k steps, `exploration_fraction=0.1` -> a 30,000-step decay window), epsilon has already fallen to roughly 0.67 by the time gradient updates begin at step 10,000, hits the 0.01 floor by step 30,000, and then **the remaining 90% of each 300k-step run trains at epsilon=0.01** (near-pure exploitation) — the 90% figure is constant regardless of budget since it's fraction-based, only the absolute step counts scale up. This is worth knowing cold: it's exactly the kind of concrete number a coach will ask for in Q&A, and it's why Lorita's `epsilon_decay` experiments (0.02 vs 0.3) directly control how much of the run is spent exploring vs exploiting.

## 6. What each hyperparameter is expected to do (hypotheses to verify against your actual runs)

- **Learning rate (`lr`)**: step size for gradient updates on the Q-network. Too high -> loss oscillates/diverges, Q-values can explode. Too low -> painfully slow convergence, may not learn anything within your timestep budget.
- **Discount factor (`gamma`)**: how much future reward matters vs. immediate reward. Pong rallies require valuing a good position several steps before scoring, so gamma too low (e.g. 0.90) tends to make the agent short-sighted / reactive rather than strategic. Gamma too close to 1 can slow convergence and increase variance in the target.
- **Batch size**: number of transitions sampled per gradient step. Larger batches -> smoother/more stable gradient estimates but more compute per update and slightly stale data (still sampled i.i.d. from the buffer). Smaller batches -> noisier updates, sometimes escapes bad local optima but can also destabilize training.
- **Buffer size**: bigger buffers hold more diverse (and older) experience, reducing correlation and catastrophic forgetting, at the cost of memory and possibly training on a very stale policy's data early on.
- **Learning starts**: purely-random warm-up length before any gradient updates. Too short -> first updates are on a tiny, low-diversity buffer. Too long -> wastes environment steps on random play.

**What we actually found, across all 30 experiments:** batch size dominated every other hyperparameter by a wide margin. Learning rate alone (0.001 or 1e-5 vs. the 0.0001 baseline) never moved the result off -21.0 at batch_size=32. But batch size showed a clean, monotonic dose-response — 16 -> -20.8, 32 -> -21.0, 64 -> -10.8, 128 -> -9.8, 256 -> +16.0 — the best result of the entire sweep, and by far. Gamma and epsilon (Lorita's block) each produced modest, real improvements (best: gamma=0.95 at -15.6, epsilon_end=0 at -14.4) but never broke into positive territory, since her block held batch_size fixed at 32. Buffer size and learning-starts (Josephine's block) also ran counter to the pre-run hypotheses: a *smaller* buffer and a *later* learning-start both outperformed the baseline, the opposite of what was predicted going in.

## 7. CNN vs MLP policy — why we compare them the way we do

- **CnnPolicy** consumes the 4-frame-stacked 84x84 pixel observation. Convolutions exploit spatial structure (paddle/ball position, motion across stacked frames) — this is the standard, high-performing setup for pixel-based Atari DQN (as in the original Mnih et al. 2015 Nature paper).
- **MlpPolicy** in this project consumes the 128-byte **RAM** state instead of pixels (`gym.make("ALE/Pong-v5", obs_type="ram")`), because feeding raw flattened pixels into a plain feedforward network throws away spatial structure and performs far worse for no principled reason — it wouldn't be a fair architecture comparison. RAM already encodes ball/paddle positions numerically in a compact vector, which is exactly the kind of input an MLP is suited for.
- **What actually happened in our runs:** MlpPolicy did not show the "faster per-step" advantage sometimes seen in theory — it simply failed to learn. All four MlpPolicy experiments (Josephine's exp2, 7, 8, 10) flatlined at exactly -21.0 reward with a near-identical, very short episode length (~757-792 steps), regardless of learning rate or batch size changes. CnnPolicy runs varied widely by comparison, several breaking past -15, with the best (batch_size=256) reaching +16.0. So for this environment and training budget, CnnPolicy on pixels was unambiguously the stronger architecture — not a close call.

## 8. Overfitting / underfitting — what they mean here, and a known bias

RL doesn't have train/test splits like supervised learning, but the same failure modes have direct analogues:
- **Underfitting** = undertrained: the Q-network hasn't had enough gradient updates to approximate the true action-values, so the policy is close to random. Expect most of the 300k-step sweep experiments to still show only modest improvement (fine for a *comparison*, not a final result) — reward staying near -21 for some configs doesn't mean the code is broken, it means 300k steps is a deliberately bounded budget. The 1M-step final model exists specifically to give the agent enough training to move well past this, and it did: mean_reward went from -21.0 at the 300k-step baseline to +10.40 at 1,000,000 steps with the winning configuration.
- **Overfitting** here looks like the policy exploiting quirks of *this specific* deterministic environment (fixed opponent behavior, no sticky actions since we use `repeat_action_probability=0.0`) rather than learning genuinely robust ball-tracking — e.g. memorizing a working action sequence for the exact frame pattern ALE produces, which is why Section 5 of the [Atari docs](https://ale.farama.org) flags sticky actions/stochastic frame-skip as a way to detect agents that memorized rather than learned. We don't use stickiness (matches the `NoFrameskip-v4` convention, chosen for fair hyperparameter comparison — see Section 1 of the README), so this is a real, known limitation worth naming if asked, not something to claim doesn't apply.
- **A separate, real bias worth knowing:** SB3's `DQN` (confirmed by reading its `train()` source) computes the TD target with `max_a' Q_target(s',a')` — this is **vanilla DQN, not Double DQN**. The max operator is a known source of *overestimation bias* (Q-values are systematically biased upward because the same network that's noisy is also the one selecting the "best" action). Double DQN fixes this by using the online network to pick the action and the target network to evaluate it. The assignment asks for plain DQN via Stable Baselines3, so this is the correct implementation choice.
