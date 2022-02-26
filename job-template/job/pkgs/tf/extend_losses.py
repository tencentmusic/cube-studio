
import tensorflow as tf


class BPRLoss(tf.keras.losses.Loss):
    def __init__(self, name="BPR_loss"):
        super(BPRLoss, self).__init__(name=name)

    @tf.function
    def call(self, y_true, y_pred):
        # [batch, 1]
        loss = tf.math.log1p(tf.math.exp(-y_pred))
        return loss


class PairHingeLoss(tf.keras.losses.Loss):
    def __init__(self, margin, name="pair_hinge_loss"):
        super(PairHingeLoss, self).__init__(name=name)
        assert margin >= 0, "'margin' must be >= 0, got {}".format(margin)
        self.margin = float(margin)

    @tf.function
    def call(self, y_true, y_pred):
        # [batch, 1]
        gap = self.margin - y_pred
        # [batch, 1]
        loss = tf.math.maximum(0., gap)
        return loss
