import csv
import random
from pathlib import Path

from gridworld_solver import (
    GridWorld,
    ORIGINAL_MAP,
    EXTENDED_MAP,
    TERMINAL,
    DISCOUNT,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
MAX_STEPS = 100


def sample_transition(world, state, action, rng):
    transitions = world.transitions(state, action)

    # Select an outcome using its probability.
    random_number = rng.random()
    cumulative_probability = 0.0

    for probability, next_state, reward in transitions:
        cumulative_probability += probability

        if random_number < cumulative_probability:
            return next_state, reward

    # Handles floating-point rounding.
    _, next_state, reward = transitions[-1]
    return next_state, reward


def run_episode(world, policy, episode_seed):
    rng = random.Random(episode_seed)

    # Find start state S.
    state = next(
        s for s in world.states
        if world.tile(s) == "S"
    )

    total_reward = 0.0
    discounted_return = 0.0
    outcome = "time_limit"
    steps = 0

    for t in range(MAX_STEPS):
        action = policy[state]

        next_state, reward = sample_transition(
            world, state, action, rng
        )

        total_reward += reward
        discounted_return += (DISCOUNT ** t) * reward

        steps += 1
        state = next_state

        if world.tile(state) in TERMINAL:
            outcome = (
                "goal" if world.tile(state) == "G"
                else "hole"
            )
            break

    return {
        "steps": steps,
        "total_reward": round(total_reward, 8),
        "discounted_return": round(discounted_return, 8),
        "outcome": outcome,
    }


def run_experiments():
    episode_rows = []
    summary_rows = []

    maps = {
        "original": ORIGINAL_MAP,
        "extended": EXTENDED_MAP,
    }

    for map_name, grid in maps.items():
        world = GridWorld(grid)

        # Compute policy from converged values.
        values, iterations, change, converged = world.solve()

        if not converged:
            raise RuntimeError(
                f"Solver did not converge: {map_name}"
            )

        policy = world.policy(values)

        print(f"\n=== {map_name.upper()} MAP ===")
        print(f"Converged in {iterations} iterations")

        for batch_seed in BATCH_SEEDS:
            batch_results = []

            for i in range(EPISODES_PER_BATCH):
                episode_seed = 1000 * batch_seed + i

                result = run_episode(
                    world, policy, episode_seed
                )

                episode_rows.append({
                    "map": map_name,
                    "batch_seed": batch_seed,
                    "episode_seed": episode_seed,
                    **result,
                })

                batch_results.append(result)

            goals = sum(
                r["outcome"] == "goal"
                for r in batch_results
            )

            holes = sum(
                r["outcome"] == "hole"
                for r in batch_results
            )

            time_limits = sum(
                r["outcome"] == "time_limit"
                for r in batch_results
            )

            mean_steps = sum(
                r["steps"] for r in batch_results
            ) / EPISODES_PER_BATCH

            mean_discounted_return = sum(
                r["discounted_return"]
                for r in batch_results
            ) / EPISODES_PER_BATCH

            summary = {
                "map": map_name,
                "batch_seed": batch_seed,
                "goals": goals,
                "holes": holes,
                "time_limits": time_limits,
                "mean_steps": round(mean_steps, 4),
                "mean_discounted_return": round(
                    mean_discounted_return, 6
                ),
            }

            summary_rows.append(summary)

            print(
                f"Batch={batch_seed} "
                f"Goals={goals} "
                f"Holes={holes} "
                f"TimeLimits={time_limits} "
                f"MeanSteps={mean_steps:.2f} "
                f"MeanReturn={mean_discounted_return:.4f}"
            )

    # Save individual episodes.
    episode_file = OUTPUT_DIR / "part_c_episodes.csv"

    with episode_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(episode_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(episode_rows)

    # Save batch summaries.
    summary_file = OUTPUT_DIR / "part_c_summary.csv"

    with summary_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved:", episode_file)
    print("Saved:", summary_file)
    print("Total episodes:", len(episode_rows))

if __name__ == "__main__":
    run_experiments()
