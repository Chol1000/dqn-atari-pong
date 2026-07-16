"""
finalize_submission.py — Fills the real results into README.md: the hyperparameter
table, a discussion draft, and a link to the recorded video.

Run after the sweep and play.py --record are done:
  python finalize_submission.py
"""
import csv
import glob
import os
import re

ROOT = os.path.dirname(__file__)
README = os.path.join(ROOT, "README.md")
RESULTS_CSV = os.path.join(ROOT, "experiments", "hyperparameter_results.csv")
CONFIG_CSV = os.path.join(ROOT, "experiments", "experiments_config.csv")
VIDEOS_DIR = os.path.join(ROOT, "videos")


def _sweep_members() -> set:
    """Real team members are whoever is listed in experiments_config.csv - this keeps
    ad-hoc runs (the sanity check, the final production model) out of the table/discussion
    without hardcoding names here."""
    if not os.path.isfile(CONFIG_CSV):
        return set()
    with open(CONFIG_CSV, newline="") as f:
        return {row["member"] for row in csv.DictReader(f)}


def _replace_between(text: str, start_marker: str, end_marker: str, new_body: str) -> str:
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
    replacement = f"{start_marker}\n{new_body}\n{end_marker}"
    if not pattern.search(text):
        raise ValueError(f"Markers {start_marker!r}/{end_marker!r} not found in README.md")
    return pattern.sub(lambda _: replacement, text, count=1)


def load_results():
    """Loads hyperparameter_results.csv, filtered to real sweep members only - excludes
    ad-hoc runs like the sanity check or the final production model (logged under
    member names that aren't in experiments_config.csv)."""
    if not os.path.isfile(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    members = _sweep_members()
    if members:
        rows = [r for r in rows if r["member"] in members]
    for r in rows:
        r["exp_id"] = int(r["exp_id"])
        r["mean_reward"] = float(r["mean_reward"])
        r["mean_ep_length"] = float(r["mean_ep_length"])
    return rows


def _baseline_comparison(rows, row) -> str:
    """Auto-computed, zero-effort fact: how this run's real numbers compare to its
    member's baseline (lowest exp_id - exp1 is always the baseline by sweep design).
    Not a substitute for the team's own analysis, but requires nobody to write anything."""
    member_rows = [r for r in rows if r["member"] == row["member"]]
    baseline = min(member_rows, key=lambda r: r["exp_id"])
    if row["exp_id"] == baseline["exp_id"]:
        return "Baseline run for this member."
    reward_delta = row["mean_reward"] - baseline["mean_reward"]
    length_delta = row["mean_ep_length"] - baseline["mean_ep_length"]
    direction = "better" if reward_delta > 0 else ("worse" if reward_delta < 0 else "no change")
    return (
        f"vs. baseline (exp{baseline['exp_id']}: {baseline['mean_reward']}): "
        f"{direction} by {abs(reward_delta):.2f} reward, "
        f"ep length {'+' if length_delta >= 0 else ''}{length_delta:.1f}"
    )


def build_table(rows) -> str:
    header = (
        "| Member | Exp # | Policy | lr | gamma | batch_size | epsilon_start | "
        "epsilon_end | epsilon_decay | Mean Reward | Mean Ep. Length | Noted Behavior |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        auto_comparison = _baseline_comparison(rows, r)
        noted_behavior = f"{r['noted_behavior']} _(auto: {auto_comparison})_"
        lines.append(
            f"| {r['member']} | {r['exp_id']} | {r['policy']} | {r['lr']} | {r['gamma']} | "
            f"{r['batch_size']} | {r['epsilon_start']} | {r['epsilon_end']} | {r['epsilon_decay']} | "
            f"{r['mean_reward']} | {r['mean_ep_length']} | {noted_behavior} |"
        )
    return "\n".join(lines)


def _describe(r: dict) -> str:
    return (
        f"**{r['member']} exp{r['exp_id']}** (`{r['policy']}`, lr={r['lr']}, gamma={r['gamma']}, "
        f"batch_size={r['batch_size']}, epsilon=({r['epsilon_start']}, {r['epsilon_end']}, {r['epsilon_decay']})) "
        f"-> mean_reward={r['mean_reward']}, mean_ep_length={r['mean_ep_length']}"
    )


def build_discussion(rows) -> str:
    if not rows:
        return "_(No results logged yet in experiments/hyperparameter_results.csv - run the sweep first.)_"

    best = max(rows, key=lambda r: r["mean_reward"])
    worst = min(rows, key=lambda r: r["mean_reward"])

    lines = [
        "_Auto-generated from `experiments/hyperparameter_results.csv` - expand with the team's own analysis:_",
        "",
        f"- **Best run overall:** {_describe(best)}",
        f"- **Worst run overall:** {_describe(worst)}",
    ]

    members = sorted({r["member"] for r in rows})
    for member in members:
        member_rows = [r for r in rows if r["member"] == member]
        member_best = max(member_rows, key=lambda r: r["mean_reward"])
        lines.append(f"- **{member}'s best run:** {_describe(member_best)}")

    return "\n".join(lines)


def build_video_link() -> str:
    clips = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")), key=os.path.getmtime)
    if not clips:
        return "_(No video found in videos/ yet - record one with `play.py --record` first.)_"
    rel_path = os.path.relpath(clips[-1], ROOT)
    return f"[{os.path.basename(rel_path)}]({rel_path})"


def main():
    with open(README) as f:
        text = f.read()

    rows = load_results()
    text = _replace_between(text, "<!-- TABLE_START -->", "<!-- TABLE_END -->", build_table(rows))
    text = _replace_between(text, "<!-- DISCUSSION_START -->", "<!-- DISCUSSION_END -->", build_discussion(rows))
    text = _replace_between(text, "<!-- VIDEO_START -->", "<!-- VIDEO_END -->", build_video_link())

    with open(README, "w") as f:
        f.write(text)

    print(f"README.md updated with {len(rows)} logged experiment(s).")
    if not rows:
        print("WARNING: no results yet - run the sweep before submitting.")
    if not glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")):
        print("WARNING: no video yet - run play.py --record before submitting.")


if __name__ == "__main__":
    main()
