from pathlib import Path

from gridworld_solver import (
    GridWorld,
    ORIGINAL_MAP,
    EXTENDED_MAP,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

original = GridWorld(ORIGINAL_MAP)
extended = GridWorld(EXTENDED_MAP)

results = []

def check(name, condition):
    assert condition, f"FAILED: {name}"
    message = f"PASS: {name}"
    results.append(message)
    print(message)

wall = original.transitions((0, 1), "D")

check(
    "Wall: agent stays on mud with reward -3",
    wall == [(1.0, (0, 1), -3)]
)


# 2. Boundary behavior
# Moving up from the upper-left corner.
boundary = original.transitions((0, 0), "U")

check(
    "Boundary: agent stays on floor with reward -1",
    boundary == [(1.0, (0, 0), -1)]
)


# 3. Ice probabilities
ice = original.transitions((1, 2), "U")

probability_sum = sum(
    probability
    for probability, state, reward in ice
)

check(
    "Ice probabilities sum to 1",
    abs(probability_sum - 1.0) < 1e-12
)

check(
    "Ice transitions use 0.8, 0.1, 0.1",
    [p for p, s, r in ice] == [0.8, 0.1, 0.1]
)

check(
    "Ice outcomes: floor, wall collision, hole",
    [(s, r) for p, s, r in ice] == [
        ((0, 2), -1),
        ((1, 2), -1),
        ((1, 3), -10),
    ]
)

mud = original.transitions((0, 0), "R")

check(
    "Mud entry reward is -3",
    mud == [(1.0, (0, 1), -3)]
)

sand = extended.transitions((2, 0), "R")

check(
    "Sand entry reward is -2",
    sand == [(1.0, (2, 1), -2)]
)

water = extended.transitions((2, 2), "R")

check(
    "Water entry reward is -5",
    water == [(1.0, (2, 3), -5)]
)

check(
    "Sand and water movement is deterministic",
    len(extended.transitions((2, 1), "R")) == 1
    and len(extended.transitions((2, 3), "L")) == 1
)

goal = original.transitions((0, 3), "L")

check(
    "Goal has no outgoing transitions or rewards",
    goal == []
)

hole = original.transitions((1, 3), "L")

check(
    "Hole has no outgoing transitions or rewards",
    hole == []
)

goal_entry = original.transitions((0, 2), "R")

check(
    "Goal entry gives +10",
    goal_entry == [(1.0, (0, 3), 10)]
)

print(f"\nALL {len(results)} RULE CHECKS PASSED")

report = "\n".join(results)
report += f"\n\nALL {len(results)} RULE CHECKS PASSED\n"

(OUTPUT_DIR / "part_c_rule_checks.txt").write_text(
    report,
    encoding="utf-8"
)

print("Saved: outputs/part_c_rule_checks.txt")
