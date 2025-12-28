import tensorflow as tf
import numpy as np


def select_action(model, obs):
    obs = np.expand_dims(obs, axis=(0, -1))  # (1, 84, 84, 1)

    logits, value = model(obs)
    probs = tf.nn.softmax(logits)

    action = tf.random.categorical(tf.math.log(probs), 1)[0, 0]
    action = int(action.numpy())

    log_prob = tf.math.log(probs[0, action] + 1e-8)

    return action, float(value[0, 0]), float(log_prob)
