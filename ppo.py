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


if __name__ == "__main__":
    logits, value = model(tf.zeros((4, 2)))
    assert logits.shape == (4, 3), logits.shape
    assert value.shape == (4, 1), value.shape

    a, logp, v = act(tf.constant([-0.5, 0.0]))
    assert a in (0, 1, 2), a
    assert logp < 0, logp                              # log of a probability is always negative

    # untrained net is near-uniform, so 300 samples should hit all 3 actions
    seen = {act(tf.constant([-0.5, 0.0]))[0] for _ in range(300)}
    assert seen == {0, 1, 2}, seen
    print("act() ok:", a, round(logp, 3), round(v, 3))
