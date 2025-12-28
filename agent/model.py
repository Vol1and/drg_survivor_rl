import tensorflow as tf
from tensorflow.keras import layers


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int):
        super().__init__()

        # --- feature extractor ---
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flatten = layers.Flatten()

        self.fc = layers.Dense(256, activation="relu")

        # --- heads ---
        self.policy_logits = layers.Dense(int(num_actions))
        self.value = layers.Dense(1)

    def call(self, x):
        """
        x: (batch, 84, 84, 1)
        """
        x = tf.cast(x, tf.float32) / 255.0

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.policy_logits(x), self.value(x)
