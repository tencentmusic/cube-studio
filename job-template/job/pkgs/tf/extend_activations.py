
import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    """
    用于DIN模型的Dice激活函数
    """
    def __init__(self, name='dice'):
        super(Dice, self).__init__(name=name)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x