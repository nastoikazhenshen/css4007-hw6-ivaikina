MAP = [
    ".M.G",
    ".#IH",
    "S..."
]

REWARDS = {
    ".": -1,
    "S": -1,
    "M": -3,
    "I": -1,
    "G": 10,
    "H": -10,
}

DISCOUNT = 0.9

ACTIONS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

SLIPS = {
    "U": ["L", "R"],
    "R": ["U", "D"],
    "D": ["L", "R"],
    "L": ["U", "D"],
}

TERMINAL = {"G", "H"}


def get_tile(state):
    row, col = state
    return MAP[row][col]


def get_states():
    states = []

    for row in range(len(MAP)):
        for col in range(len(MAP[row])):
            if MAP[row][col] != "#":
                states.append((row, col))

    return states


def move(state, action):
    row, col = state
    dr, dc = ACTIONS[action]

    new_row = row + dr
    new_col = col + dc

    if (
        new_row < 0
        or new_row >= len(MAP)
        or new_col < 0
        or new_col >= len(MAP[new_row])
        or MAP[new_row][new_col] == "#"
    ):
        return state

    return (new_row, new_col)


def get_transitions(state, action):
    tile = get_tile(state)

    # Terminal states have no more actions
    if tile in TERMINAL:
        return []

    # Ice: 80% intended, 10% each perpendicular
    if tile == "I":
        directions = [
            (action, 0.8),
            (SLIPS[action][0], 0.1),
            (SLIPS[action][1], 0.1),
        ]
    else:
        directions = [(action, 1.0)]

    transitions = []

    for direction, probability in directions:
        next_state = move(state, direction)
        reward = REWARDS[get_tile(next_state)]

        transitions.append(
            (probability, next_state, reward)
        )

    return transitions


def action_value(state, action, values):
    total = 0.0

    for probability, next_state, reward in get_transitions(
        state, action
    ):
        total += probability * (
            reward + DISCOUNT * values[next_state]
        )

    return total


def value_iteration_step(values, states):
    new_values = {}

    for state in states:
        if get_tile(state) in TERMINAL:
            new_values[state] = 0.0
            continue

        action_values = [
            action_value(state, action, values)
            for action in ACTIONS
        ]

        new_values[state] = max(action_values)

    return new_values


def print_values(values, title):
    print("\n" + title)

    for row in range(len(MAP)):
        cells = []

        for col in range(len(MAP[row])):
            state = (row, col)

            if MAP[row][col] == "#":
                cells.append("   #   ")
            else:
                cells.append(f"{values[state]:7.2f}")

        print(" ".join(cells))


def main():
    states = get_states()

    # V0: all values start at zero
    v0 = {state: 0.0 for state in states}

    # First iteration
    v1 = value_iteration_step(v0, states)

    # Second iteration uses V1
    v2 = value_iteration_step(v1, states)

    print_values(v0, "V0 — Initial Values")
    print_values(v1, "V1 — First Iteration")

    print("\nV2 — Requested States")

    requested_states = [
        (0, 1),  
        (0, 2),
        (1, 2),
    ]

    for state in requested_states:
        paper_row = state[0] + 1
        paper_col = state[1] + 1

        print(
            f"State ({paper_row},{paper_col}): "
            f"V2 = {v2[state]:.2f}"
        )

    # Compare program results with paper calculations
    expected_v1 = [
        [-1, -1, 10, 0],
        [-1, None, -1, 0],
        [-1, -1, -1, -1],
    ]

    for row in range(len(MAP)):
        for col in range(len(MAP[row])):
            expected = expected_v1[row][col]

            if expected is not None:
                assert abs(v1[(row, col)] - expected) < 1e-9

    assert abs(v2[(0, 1)] - 8.0) < 1e-9
    assert abs(v2[(0, 2)] - 10.0) < 1e-9
    assert abs(v2[(1, 2)] - 5.21) < 1e-9

    print("\nALL PART A CHECKS PASSED")


if __name__ == "__main__":
    main()
