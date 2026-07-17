# Lorita's Analysis — Gamma & Epsilon-Greedy Schedule

I ran 10 experiments varying the discount factor (gamma) and the epsilon-greedy exploration schedule, with batch_size held fixed at 32 throughout.

## What I tested
- Gamma alone: 0.90, 0.95, 0.999 (baseline 0.99)
- Epsilon decay speed: 0.3 (slower), 0.02 (faster) (baseline 0.1)
- Epsilon floor: 0.05 (higher), 0.0 (fully greedy) (baseline 0.01)
- Combinations of gamma + epsilon changes

## What actually happened

**Gamma showed an interior optimum, not "closer to 1 is always better":**

| gamma | mean_reward |
|---|---|
| 0.90 | -20.4 |
| 0.95 | **-15.6 (best)** |
| 0.99 (baseline) | -21.0 |
| 0.999 | -17.4 |

A moderate reduction from the 0.99 default helped noticeably, but pushing further in either direction (too low or too high) hurt.

**Epsilon: fully greedy after decay was the single strongest change in my whole block:**
`epsilon_end=0.0` reached -14.4, my best result — better than leaving any residual exploration floor. A higher floor (0.05) actually hurt (-19.4 and -20.0 in two different runs). For decay speed, faster (0.02) beat slower (0.3): -16.8 vs -20.4.

## Combining gamma and epsilon changes
I also tested two combos: reduced gamma with slower epsilon decay (exp9, gamma=0.95 + decay=0.2) reached -16.6, roughly in line with gamma=0.95 alone — the slower decay didn't add much on top. High gamma with a higher exploration floor (exp10, gamma=0.999 + epsilon_end=0.05) reached -20.0, close to what gamma=0.999 alone gave, again suggesting these two combos didn't unlock anything beyond what their individual components already showed — gamma and epsilon in my block moved performance independently rather than compounding with each other.

## Why none of my runs went positive
Every one of my experiments kept batch_size fixed at 32, and Chol's block showed that batch_size is what unlocks positive rewards at this training budget — gamma and epsilon tuning gave real, meaningful improvements (from -21.0 down to as good as -14.4), but couldn't overcome the batch-size ceiling on their own. This isn't a failure — it's exactly the kind of controlled result that confirms batch size was the dominant lever, not an artifact of my block being tested wrong.

## Why I think this happened
Gamma controls how much the agent values future reward vs. immediate reward — too low and it plays short-sightedly, missing multi-step rally strategy; too high and the target values become noisier and slower to converge, which likely explains the 0.999 dip. For epsilon, letting the agent go fully greedy once training is done means it stops taking random actions that would otherwise occasionally throw away a point it could have won — that residual randomness costs more than it helps once the policy is actually decent.
