import csv
import random
from pathlib import Path

import gymnasium as gym

from fixed_policy import FIXED_POLICY


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EPISODES_FILE = OUTPUT_DIR / "part_b_episodes.csv"

GOAL_STATE = 15
HOLE_STATES = {5, 7, 11, 12}
MAX_STEPS = 100

ACTION_NAMES = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
}


def find_example(outcome):
    with EPISODES_FILE.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if (
            row["slippery"] == "True"
            and row["policy"] == "fixed"
            and row["outcome"] == outcome
        ):
            return row

    raise ValueError(f"No example found: {outcome}")


def save_trace(row, filename):
    seed = int(row["episode_seed"])
    slippery = row["slippery"] == "True"
    policy = row["policy"]

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=slippery,
        render_mode="ansi",
    )

    state, _ = env.reset(seed=seed)
    rng = random.Random(seed)

    lines = [
        "FrozenLake-v1 — Episode Trace",
        f"Policy: {policy}",
        f"Slippery: {slippery}",
        f"Batch seed: {row['batch_seed']}",
        f"Episode seed: {seed}",
        "",
        "Initial map:",
        env.render(),
    ]

    total_reward = 0.0
    outcome = "time_limit"
    steps = 0

    for step in range(1, MAX_STEPS + 1):
        if policy == "fixed":
            action = FIXED_POLICY[state]
        else:
            action = rng.randrange(4)

        previous_state = state

        state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        total_reward += float(reward)
        steps = step

        lines.extend([
            f"Step {step}",
            f"From state: {previous_state}",
            f"Chosen action: {ACTION_NAMES[action]}",
            f"To state: {state}",
            f"Reward: {reward}",
            env.render(),
        ])

        if terminated:
            if state == GOAL_STATE:
                outcome = "goal"
            elif state in HOLE_STATES:
                outcome = "hole"
            else:
                raise ValueError(
                    f"Unknown terminal state: {state}"
                )
            break

        if truncated:
            outcome = "time_limit"
            break

    lines.extend([
        "FINAL RESULT",
        f"Outcome: {outcome}",
        f"Steps: {steps}",
        f"Total reward: {total_reward}",
    ])

    # Verify this trace reproduces the saved experiment.
    assert outcome == row["outcome"]
    assert steps == int(row["steps"])
    assert total_reward == float(row["total_reward"])

    path = OUTPUT_DIR / filename
    path.write_text("\n".join(lines), encoding="utf-8")

    env.close()

    print(
        f"Saved {path} | "
        f"seed={seed} | "
        f"outcome={outcome} | "
        f"steps={steps} | "
        f"reward={total_reward}"
    )


def main():
    success = find_example("goal")
    failure = find_example("hole")

    save_trace(success, "part_b_success_trace.txt")
    save_trace(failure, "part_b_failure_trace.txt")


if __name__ == "__main__":
    main()
