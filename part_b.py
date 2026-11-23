
import gymnasium as gym

# Create the FrozenLake environment
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=False,
    render_mode="ansi"
)

# Start a new episode
state, info = env.reset(seed=11)

print("INITIAL MAP:")
print(env.render())

# FrozenLake actions:
# 0 = Left, 1 = Down, 2 = Right, 3 = Up
actions = [1, 1, 2, 2, 1, 2]

total_reward = 0
steps = 0

for action in actions:
    state, reward, terminated, truncated, info = env.step(action)

    steps += 1
    total_reward += reward

    print(f"\nStep {steps}")
    print(f"Action: {action}")
    print(f"State: {state}")
    print(f"Reward: {reward}")
    print(env.render())

    if terminated or truncated or steps >= 100:
        break

print("\nEPISODE RESULT")
print(f"Steps: {steps}")
print(f"Total reward: {total_reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

env.close()
