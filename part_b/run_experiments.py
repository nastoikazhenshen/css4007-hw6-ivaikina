import csv
import random
from pathlib import Path

import gymnasium as gym

from fixed_policy import FIXED_POLICY


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
MAX_STEPS = 100

GOAL_STATE = 15
HOLE_STATES = {5, 7, 11, 12}


def run_episode(env, policy, episode_seed):
    state, _ = env.reset(seed=episode_seed)

    # Separate, reproducible random action generator
    rng = random.Random(episode_seed)

    total_reward = 0.0
    steps = 0
    outcome = "time_limit"

    for step in range(MAX_STEPS):
        if policy == "random":
            action = rng.randrange(4)
        else:
            action = FIXED_POLICY[state]

        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)
        steps += 1
        state = next_state

        if terminated:
            if state == GOAL_STATE:
                outcome = "goal"
            elif state in HOLE_STATES:
                outcome = "hole"
            else:
                raise ValueError(f"Unknown terminal state: {state}")
            break

        if truncated:
            outcome = "time_limit"
            break

    return {
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def run_all_experiments():
    episode_rows = []
    summary_rows = []

    for slippery in [False, True]:
        for policy in ["random", "fixed"]:
            env = gym.make(
                "FrozenLake-v1",
                map_name="4x4",
                is_slippery=slippery,
            )

            for batch_seed in BATCH_SEEDS:
                batch_results = []

                for i in range(EPISODES_PER_BATCH):
                    episode_seed = 1000 * batch_seed + i

                    result = run_episode(env, policy, episode_seed)

                    episode_rows.append({
                        "condition": f"slippery_{slippery}_{policy}",
                        "slippery": slippery,
                        "batch_seed": batch_seed,
                        "episode_seed": episode_seed,
                        "policy": policy,
                        **result,
                    })

                    batch_results.append(result)

                goals = sum(r["outcome"] == "goal" for r in batch_results)
                holes = sum(r["outcome"] == "hole" for r in batch_results)
                time_limits = sum(
                    r["outcome"] == "time_limit"
                    for r in batch_results
                )

                mean_steps = sum(
                    r["steps"] for r in batch_results
                ) / EPISODES_PER_BATCH

                summary = {
                    "slippery": slippery,
                    "policy": policy,
                    "batch_seed": batch_seed,
                    "goals": goals,
                    "holes": holes,
                    "time_limits": time_limits,
                    "success_percent": 100 * goals / EPISODES_PER_BATCH,
                    "mean_steps": round(mean_steps, 2),
                }

                summary_rows.append(summary)

                print(
                    f"Slippery={str(slippery):5s} "
                    f"Policy={policy:6s} "
                    f"Batch={batch_seed:2d} "
                    f"Goals={goals:2d} "
                    f"Holes={holes:2d} "
                    f"TimeLimits={time_limits:2d} "
                    f"Success={summary['success_percent']:5.1f}% "
                    f"MeanSteps={mean_steps:.2f}"
                )

            env.close()

    episode_file = OUTPUT_DIR / "part_b_episodes.csv"
    with episode_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(episode_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_file = OUTPUT_DIR / "part_b_summary.csv"
    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:", episode_file)
    print("Saved:", summary_file)
    print("Total episodes:", len(episode_rows))


if __name__ == "__main__":
    run_all_experiments()
