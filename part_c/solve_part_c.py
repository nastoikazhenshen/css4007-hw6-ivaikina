from pathlib import Path

from value_iteration import (
    MAP,
    ACTIONS,
    TERMINAL,
    get_states,
    get_tile,
    action_value,
    value_iteration_step,
    print_values,
)

TOLERANCE = 0.000001
MAX_ITERATIONS = 10000

ARROWS = {
    "U": "↑",
    "R": "→",
    "D": "↓",
    "L": "←",
}


def solve():
    states = get_states()
    values = {state: 0.0 for state in states}

    converged = False
    final_change = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = value_iteration_step(values, states)

        final_change = max(
            abs(new_values[state] - values[state])
            for state in states
        )

        values = new_values

        if final_change < TOLERANCE:
            converged = True
            break

    return values, iteration, final_change, converged


def extract_policy(values):
    policy = {}

    for state in get_states():
        if get_tile(state) in TERMINAL:
            continue

        scores = {
            action: action_value(state, action, values)
            for action in ACTIONS
        }

        best_value = max(scores.values())

        best_action = next(
            action for action in ACTIONS
            if abs(scores[action] - best_value) < 1e-10
        )

        policy[state] = best_action

    return policy

def format_policy(policy):
    lines = []

    for row in range(len(MAP)):
        cells = []

        for col in range(len(MAP[row])):
            tile = MAP[row][col]
            state = (row, col)

            if tile in TERMINAL or tile == "#":
                symbol = tile
            else:
                symbol = ARROWS[policy[state]]

            cells.append(f"{symbol:^7}")

        lines.append(" ".join(cells))

    return "\n".join(lines)


def main():
    values, iterations, change, converged = solve()

    print("PART C2 — VALUE ITERATION")
    print("Map: Original Part A")
    print(f"Tolerance: {TOLERANCE}")
    print(f"Maximum iterations: {MAX_ITERATIONS}")

    print(f"\nConverged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final maximum change: {change:.10f}")

    if not converged:
        print("WARNING: Solver did not converge.")
        return

    print_values(values, "FINAL VALUE MAP")

    policy = extract_policy(values)
    policy_text = format_policy(policy)

    print("\nFINAL OPTIMAL POLICY")
    print(policy_text)

    print("\nTie-breaking rule: Up, Right, Down, Left")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    value_lines = []
    for row in range(len(MAP)):
        cells = []

        for col in range(len(MAP[row])):
            state = (row, col)

            if MAP[row][col] == "#":
                cells.append("   #   ")
            else:
                cells.append(f"{values[state]:7.4f}")

        value_lines.append(" ".join(cells))

    (output_dir / "part_c_original_values.txt").write_text(
        "\n".join(value_lines) + "\n",
        encoding="utf-8",
    )

    (output_dir / "part_c_original_policy.txt").write_text(
        policy_text + "\n",
        encoding="utf-8",
    )

    print("\nSaved final value map and policy map.")


if __name__ == "__main__":
    main()
