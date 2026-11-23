import gymnasium as gym
from fixed_policy import FIXED_POLICY

env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False
)

state, info = env.reset(seed=11)

print("Starting state:", state)

for step in range(1, 101):
    action = FIXED_POLICY[state]

    state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step}: "
        f"action={action}, "
        f"state={state}, "
        f"reward={reward}"
    )

    if terminated or truncated:
        print("Final outcome:", "Goal" if state == 15 else "Hole")
        break
else:
    print("Final outcome: Time limit")

env.close()
