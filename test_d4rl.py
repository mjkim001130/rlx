import gym, d4rl  # noqa: F401

env = gym.make("hopper-medium-v2")
print("env:", env)
obs = env.reset()
for _ in range(5):
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
ds = env.get_dataset()
print("dataset keys:", list(ds.keys())[:5])
print("obs/actions:", ds["observations"].shape, ds["actions"].shape)
