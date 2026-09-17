import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("MountainCar-v0", render_mode="human")


model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)

model.save("mountain_car_ppo")
env.close()