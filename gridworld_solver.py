from pathlib import Path

ORIGINAL_MAP = [
    ".M.G",
    ".#IH",
    "S...",
]

EXTENDED_MAP = [
    ".M.G",
    ".#IH",
    "SD.W",
]

REWARDS = {
    ".": -1,
    "S": -1,
    "M": -3,
    "I": -1,
    "D": -2,
    "W": -5,
    "G": 10,
    "H": -10,
}

ACTIONS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

SLIPS = {
    "U": ("L", "R"),
    "R": ("U", "D"),
    "D": ("L", "R"),
    "L": ("U", "D"),
}

ARROWS = {
    "U": "↑",
    "R": "→",
    "D": "↓",
    "L": "←",
}

TERMINAL = {"G", "H"}
DISCOUNT = 0.9
TOLERANCE = 0.000001
MAX_ITERATIONS = 10000


class GridWorld:
    def __init__(self, grid, rewards=REWARDS):
        self.grid = grid
        self.rewards = rewards
        self.states = [
            (r, c)
            for r, line in enumerate(grid)
            for c, tile in enumerate(line)
            if tile != "#"
        ]

    def tile(self, state):
        r, c = state
        return self.grid[r][c]

    def move(self, state, action):
        r, c = state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        if (
            nr < 0 or nr >= len(self.grid)
            or nc < 0 or nc >= len(self.grid[nr])
            or self.grid[nr][nc] == "#"
        ):
            return state

        return (nr, nc)

    def transitions(self, state, action):
        if self.tile(state) in TERMINAL:
            return []

        if self.tile(state) == "I":
            directions = [
                (action, 0.8),
                (SLIPS[action][0], 0.1),
                (SLIPS[action][1], 0.1),
            ]
        else:
            directions = [(action, 1.0)]

        result = []

        for direction, probability in directions:
            next_state = self.move(state, direction)
            reward = self.rewards[self.tile(next_state)]
            result.append((probability, next_state, reward))

        return result

    def action_value(self, state, action, values):
        return sum(
            probability * (
                reward + DISCOUNT * values[next_state]
            )
            for probability, next_state, reward
            in self.transitions(state, action)
        )

    def update(self, values):
        new_values = {}

        for state in self.states:
            if self.tile(state) in TERMINAL:
                new_values[state] = 0.0
            else:
                new_values[state] = max(
                    self.action_value(state, action, values)
                    for action in ACTIONS
                )

        return new_values

    def solve(self):
        values = {state: 0.0 for state in self.states}
        converged = False

        for iteration in range(1, MAX_ITERATIONS + 1):
            new_values = self.update(values)

            change = max(
                abs(new_values[s] - values[s])
                for s in self.states
            )

            values = new_values

            if change < TOLERANCE:
                converged = True
                break

        return values, iteration, change, converged

    def policy(self, values):
        result = {}

        for state in self.states:
            if self.tile(state) in TERMINAL:
                continue

            scores = {
                a: self.action_value(state, a, values)
                for a in ACTIONS
            }

            best = max(scores.values())

            # Tie-breaking order: U, R, D, L
            result[state] = next(
                a for a in ACTIONS
                if abs(scores[a] - best) < 1e-10
            )

        return result

    def format_values(self, values):
        lines = []

        for r, line in enumerate(self.grid):
            cells = []
            for c, tile in enumerate(line):
                if tile == "#":
                    cells.append("   #   ")
                else:
                    cells.append(f"{values[(r, c)]:7.4f}")
            lines.append(" ".join(cells))

        return "\n".join(lines)

    def format_policy(self, policy):
        lines = []

        for r, line in enumerate(self.grid):
            cells = []
            for c, tile in enumerate(line):
                if tile == "#" or tile in TERMINAL:
                    symbol = tile
                else:
                    symbol = ARROWS[policy[(r, c)]]

                cells.append(f"{symbol:^7}")

            lines.append(" ".join(cells))

        return "\n".join(lines)


def run_map(name, grid):
    world = GridWorld(grid)

    values, iterations, change, converged = world.solve()

    print(f"\n=== {name} ===")
    print(f"Converged: {converged}")
    print(f"Iterations: {iterations}")
    print(f"Final maximum change: {change:.10f}")

    if not converged:
        print("WARNING: Did not converge.")
        return

    policy = world.policy(values)

    value_text = world.format_values(values)
    policy_text = world.format_policy(policy)

    print("\nFINAL VALUES")
    print(value_text)

    print("\nOPTIMAL POLICY")
    print(policy_text)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    prefix = f"part_c_{name.lower()}"

    (output_dir / f"{prefix}_values.txt").write_text(
        value_text + "\n", encoding="utf-8"
    )

    (output_dir / f"{prefix}_policy.txt").write_text(
        policy_text + "\n", encoding="utf-8"
    )

    print("\nSaved values and policy maps.")


if __name__ == "__main__":
    run_map("original", ORIGINAL_MAP)
    run_map("extended", EXTENDED_MAP)
