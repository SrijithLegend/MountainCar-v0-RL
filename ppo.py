import sys

import gymnasium as gym
import numpy as np
import tensorflow as tf

ENV_ID = "CartPole-v1"          # swap to "MountainCar-v0" once the loop is proven

env = gym.make(ENV_ID)
OBS_DIM = env.observation_space.shape[0]
N_ACTIONS = int(env.action_space.n)   # numpy int; keras wants a plain int

obs = tf.keras.layers.Input(shape=(OBS_DIM,))

policyNN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(OBS_DIM,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(N_ACTIONS)
])


valueNN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(OBS_DIM,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])


# two parallel heads off the same observation, not a chain
model = tf.keras.Model(obs, [policyNN(obs), valueNN(obs)])

optimizer = tf.keras.optimizers.Adam(3e-4)

model.summary()


def act(observation):
    """One env step's worth: sample an action, report its log-prob and the critic's value."""
    logits, value = model(observation[None])          # add batch dim: (D,) -> (1, D)
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
        buf["terminated"].append(terminated)   # real ending: no future reward
        buf["truncated"].append(truncated)     # time limit: future reward still exists

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # obs is handed back so the next rollout continues the episode instead of restarting it
    return {k: np.array(v) for k, v in buf.items()}, obs


def advantages(batch, gamma=0.99, lam=0.95):
    """GAE: how much better each action turned out than the critic expected."""
    rewards, values = batch["rewards"], batch["values"]

    # critic's guess for the state each step landed in; zero once the episode really ended
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


# ponytail: eager, not @tf.function - ~3x slower but no retracing surprises.
# Wrap it once the training loop is the bottleneck.
def train_step(obs, actions, old_logps, adv, returns,
               clip=0.2, vf_coef=0.5, ent_coef=0.01):
    """One gradient update on a minibatch: clipped policy loss + value loss + entropy bonus."""
    obs = tf.cast(obs, tf.float32)
    actions = tf.cast(actions, tf.int32)
    old_logps, adv, returns = (tf.cast(x, tf.float32) for x in (old_logps, adv, returns))

    with tf.GradientTape() as tape:
        logits, values = model(obs)
        logp_all = tf.nn.log_softmax(logits)
        new_logps = tf.gather(logp_all, actions, batch_dims=1)

        # how much more (or less) likely this action is now than when it was collected
        ratio = tf.exp(new_logps - old_logps)
        clipped = tf.clip_by_value(ratio, 1 - clip, 1 + clip)
        # the pessimistic one: caps the gain from any single update
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped * adv))

        value_loss = tf.reduce_mean(tf.square(values[:, 0] - returns))
        entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1))

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 0.5)     # one more brake on runaway updates
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(policy_loss), float(value_loss), float(entropy)


def train(env, iterations=40, n_steps=1024, epochs=10, batch_size=64):
    """collect -> score -> update, over and over. Prints the average episode return."""
    obs, history = None, []

    for i in range(iterations):
        batch, obs = collect(env, n_steps, obs)
        adv, returns = advantages(batch)

        idx = np.arange(n_steps)
        for _ in range(epochs):                      # reuse each rollout ~10x; the clip makes that safe
            np.random.shuffle(idx)                   # shuffle so minibatches aren't correlated in time
            for start in range(0, n_steps, batch_size):
                mb = idx[start:start + batch_size]
                pl, vl, ent = train_step(batch["obs"][mb], batch["actions"][mb],
                                         batch["logps"][mb], adv[mb], returns[mb])

        # approximate: ignores the partial episode at each end of the rollout
        finished = int((batch["terminated"] | batch["truncated"]).sum())
        ep_return = batch["rewards"].sum() / max(finished, 1)
        history.append(ep_return)
        print(f"iter {i:3d} | return {ep_return:8.1f} | value loss {vl:8.2f} | entropy {ent:.3f}",
              flush=True)

    return history


if __name__ == "__main__":
    if "--train" in sys.argv:
        train(env)
        env.close()
        sys.exit()

    logits, value = model(tf.zeros((4, OBS_DIM)))
    assert logits.shape == (4, N_ACTIONS), logits.shape
    assert value.shape == (4, 1), value.shape

    start_obs, _ = env.reset(seed=0)
    a, logp, v = act(start_obs)
    assert a in range(N_ACTIONS), a
    assert logp < 0, logp                              # log of a probability is always negative

    # untrained net is near-uniform, so 300 samples should hit every action
    seen = {act(start_obs)[0] for _ in range(300)}
    assert seen == set(range(N_ACTIONS)), seen
    print("act() ok:", a, round(logp, 3), round(v, 3))

    batch, last_obs = collect(env, n_steps=250)

    assert batch["obs"].shape == (250, OBS_DIM), batch["obs"].shape
    assert batch["actions"].shape == (250,), batch["actions"].shape
    assert np.isfinite(batch["rewards"]).all()
    assert (batch["terminated"] | batch["truncated"]).sum() >= 1   # 250 steps must end some episode
    assert last_obs.shape == (OBS_DIM,), last_obs.shape
    print("collect() ok:", len(batch["obs"]), "steps,",
          int((batch["terminated"] | batch["truncated"]).sum()), "episodes finished")

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

    args = (batch["obs"], batch["actions"], batch["logps"], adv, returns)

    # before any update the weights are unchanged, so every ratio is exactly 1
    # and the policy loss collapses to -mean(adv), which normalization made 0
    pl, vl, ent = train_step(*args)
    assert abs(pl) < 1e-5, pl
    assert abs(ent - np.log(N_ACTIONS)) < 0.05, ent   # untrained policy is near-uniform

    for _ in range(30):
        pl, vl2, ent = train_step(*args)
    assert vl2 < vl, (vl, vl2)                        # critic must fit the returns it was handed
    print("train_step() ok: value loss", round(vl, 3), "->", round(vl2, 3),
          "| entropy", round(ent, 3))
    env.close()
    print(f"all checks pass on {ENV_ID}. run `python ppo.py --train` to train.")
