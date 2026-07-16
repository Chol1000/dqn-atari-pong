"""
run_experiments.py — Runs the hyperparameter sweep from experiments_config.csv
and rebuilds the results table.

Usage:
  python run_experiments.py                       # run everything
  python run_experiments.py --member "Chol"        # one member's 10 experiments
  python run_experiments.py --exp-id 3 --member "Lorita"  # a single experiment
  python run_experiments.py --generate-table-only  # just rebuild the table
"""
import argparse
import csv
import os
from types import SimpleNamespace

from train import train, RESULTS_CSV

CONFIG_CSV = os.path.join(os.path.dirname(__file__), "experiments", "experiments_config.csv")
TABLE_MD = os.path.join(os.path.dirname(__file__), "experiments", "hyperparameter_table.md")

# Fields not present in experiments_config.csv - shared across every run in the sweep.
DEFAULTS = dict(
    env="Pong",
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    n_envs=1,
    seed=0,
    eval_freq=10_000,
    eval_episodes=5,
    progress_bar=False,
    checkpoint_freq=0,  # sweep experiments are already resumable per-experiment (idempotent skip), no mid-run checkpoint needed
    tensorboard_log="tensorboard_logs",
)


def load_config():
    with open(CONFIG_CSV, newline="") as f:
        return list(csv.DictReader(f))


def completed_keys() -> set:
    """(member, exp_id) pairs already logged in RESULTS_CSV - lets the sweep be safely
    re-run (e.g. after a Colab disconnect) without retraining or duplicating rows."""
    if not os.path.isfile(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, newline="") as f:
        return {(row["member"], int(row["exp_id"])) for row in csv.DictReader(f)}


def row_to_args(row: dict) -> SimpleNamespace:
    member_slug = row["member"].replace(" ", "_")
    model_path = f"models/{member_slug}_exp{row['exp_id']}_{row['policy']}.zip"
    ns = dict(DEFAULTS)
    ns.update(
        member=row["member"],
        exp_id=int(row["exp_id"]),
        policy=row["policy"],
        lr=float(row["lr"]),
        gamma=float(row["gamma"]),
        batch_size=int(row["batch_size"]),
        epsilon_start=float(row["epsilon_start"]),
        epsilon_end=float(row["epsilon_end"]),
        epsilon_decay=float(row["epsilon_decay"]),
        buffer_size=int(row["buffer_size"]),
        learning_starts=int(row["learning_starts"]),
        total_timesteps=int(row["total_timesteps"]),
        notes=row.get("hypothesis", ""),
        model_path=model_path,
    )
    return SimpleNamespace(**ns)


def generate_table():
    if not os.path.isfile(RESULTS_CSV):
        print(f"No results yet at {RESULTS_CSV} — run some experiments first.")
        return
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    header = (
        "| Member | Exp # | Policy | lr | gamma | batch_size | epsilon_start | "
        "epsilon_end | epsilon_decay | Mean Reward | Mean Ep. Length | Noted Behavior |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['member']} | {r['exp_id']} | {r['policy']} | {r['lr']} | {r['gamma']} | "
            f"{r['batch_size']} | {r['epsilon_start']} | {r['epsilon_end']} | {r['epsilon_decay']} | "
            f"{r['mean_reward']} | {r['mean_ep_length']} | {r['noted_behavior']} |\n"
        )
    with open(TABLE_MD, "w") as f:
        f.writelines(lines)
    print(f"Wrote {TABLE_MD} ({len(rows)} rows). Paste its contents into README.md.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--member", type=str, default=None, help="Only run this member's experiments")
    p.add_argument("--exp-id", type=int, default=None, help="Only run this experiment number")
    p.add_argument("--generate-table-only", action="store_true")
    args = p.parse_args()

    if args.generate_table_only:
        generate_table()
        return

    rows = load_config()
    if args.member:
        rows = [r for r in rows if r["member"] == args.member]
    if args.exp_id:
        rows = [r for r in rows if int(r["exp_id"]) == args.exp_id]

    if not rows:
        print("No matching experiments found — check --member/--exp-id filters.")
        return

    done = completed_keys()
    to_run = [r for r in rows if (r["member"], int(r["exp_id"])) not in done]
    skipped = len(rows) - len(to_run)
    if skipped:
        print(f"Skipping {skipped} experiment(s) already completed in a previous run.")

    print(f"Running {len(to_run)} experiment(s)...")
    for row in to_run:
        run_args = row_to_args(row)
        print(f"\n=== {run_args.member} / exp{run_args.exp_id} / {run_args.policy} ===")
        train(run_args)

    generate_table()


if __name__ == "__main__":
    main()
