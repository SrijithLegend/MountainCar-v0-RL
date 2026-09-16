import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

obs, info = env.reset()
print('X-axis position: ', obs[0])
print('Initial velocity: ', obs[1])