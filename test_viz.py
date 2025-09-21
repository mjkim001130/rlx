import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import gym, d4rl

env = gym.make("hopper-medium-v2")
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()

env.close()
