import numpy as np
import tensorflow as tf

obs = tf.keras.layers.Input(shape=(2,))

policyNN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3)
])
policyNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")


valueNN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])
valueNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")


# two parallel heads off the same observation, not a chain
model = tf.keras.Model(obs, [policyNN(obs), valueNN(obs)])

model.summary()


def act(observation):
    """One env step's worth: sample an action, report its log-prob and the critic's value."""
    logits, value = model(observation[None])          # add batch dim: (2,) -> (1, 2)
    action = int(tf.random.categorical(logits, 1)[0, 0])  # samples from softmax(logits)
    logp = float(tf.nn.log_softmax(logits)[0, action])
    return action, logp, float(value[0, 0])


def collect(env, n_steps=2048, obs=None):
    """Run the current policy for n_steps, recording everything the PPO update needs."""
    if obs is None:
        obs, _ = env.reset()

    buf = {k: [] for k in
           ("obs", "actions", "logps", "values", "rewards", "terminated", "truncated")}

    for _ in range(n_steps):
        action, logp, value = act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["logps"].append(logp)
        buf["values"].append(value)
        buf["rewards"].append(reward)
        buf["terminated"].append(terminated)   # reached the flag: no future reward
        buf["truncated"].append(truncated)     # 200-step cutoff: future reward still exists

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # obs is handed back so the next rollout continues the episode instead of restarting it
    return {k: np.array(v) for k, v in buf.items()}, obs


if __name__ == "__main__":
    logits, value = model(tf.zeros((4, 2)))
    assert logits.shape == (4, 3), logits.shape
    assert value.shape == (4, 1), value.shape

    a, logp, v = act(np.array([-0.5, 0.0], dtype=np.float32))
    assert a in (0, 1, 2), a
    assert logp < 0, logp                              # log of a probability is always negative

    # untrained net is near-uniform, so 300 samples should hit all 3 actions
    seen = {act(np.array([-0.5, 0.0], dtype=np.float32))[0] for _ in range(300)}
    assert seen == {0, 1, 2}, seen
    print("act() ok:", a, round(logp, 3), round(v, 3))

    import gymnasium as gym

    env = gym.make("MountainCar-v0")
    batch, last_obs = collect(env, n_steps=250)
    env.close()

    assert batch["obs"].shape == (250, 2), batch["obs"].shape
    assert batch["actions"].shape == (250,), batch["actions"].shape
    assert set(np.unique(batch["rewards"])) == {-1.0}   # MountainCar pays -1 per step, always
    assert batch["truncated"].sum() >= 1                # 250 steps must cross the 200-step cutoff
    assert last_obs.shape == (2,), last_obs.shape
    print("collect() ok:", len(batch["obs"]), "steps,",
          int(batch["terminated"].sum()), "reached the flag,",
          int(batch["truncated"].sum()), "timed out")
