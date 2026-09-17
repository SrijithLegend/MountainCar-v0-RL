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
           ("obs", "next_obs", "actions", "logps", "values", "rewards", "terminated", "truncated")}

    for _ in range(n_steps):
        action, logp, value = act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        buf["obs"].append(obs)
        buf["next_obs"].append(next_obs)   # the real landing spot, not the post-reset one
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


def advantages(batch, gamma=0.99, lam=0.95):
    """GAE: how much better each action turned out than the critic expected."""
    rewards, values = batch["rewards"], batch["values"]

    # critic's guess for the state each step landed in; zero once the flag is reached
    next_values = model(batch["next_obs"])[1][:, 0].numpy()
    next_values = np.where(batch["terminated"], 0.0, next_values)

    # surprise at each step: what we actually got vs. what we expected to get
    deltas = rewards + gamma * next_values - values

    adv = np.zeros_like(deltas)
    running = 0.0
    for t in reversed(range(len(deltas))):           # backwards: the future is known first
        if batch["terminated"][t] or batch["truncated"][t]:
            running = 0.0                            # never blend across an episode boundary
        running = deltas[t] + gamma * lam * running
        adv[t] = running

    returns = adv + values                           # the critic's training target
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)    # normalize: keeps updates from exploding
    return adv.astype(np.float32), returns.astype(np.float32)


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

    adv, returns = advantages(batch)
    assert adv.shape == returns.shape == (250,), adv.shape
    assert np.isfinite(adv).all() and np.isfinite(returns).all()
    assert abs(adv.mean()) < 1e-5 and abs(adv.std() - 1) < 1e-5

    # lam=0 makes GAE collapse to the one-step surprise, which we can recompute independently
    nv = np.where(batch["terminated"], 0.0, model(batch["next_obs"])[1][:, 0].numpy())
    deltas = batch["rewards"] + 0.99 * nv - batch["values"]
    expected = (deltas - deltas.mean()) / (deltas.std() + 1e-8)
    assert np.allclose(advantages(batch, lam=0.0)[0], expected, atol=1e-5)
    print("advantages() ok: mean", round(float(adv.mean()), 6),
          "std", round(float(adv.std()), 4),
          "| returns range", round(float(returns.min()), 2), "to", round(float(returns.max()), 2))
