import tensorflow as tf
from deepctr.layers.activation import activation_layer
from deepctr.layers.utils import reduce_max, reduce_mean, reduce_sum, concat_func, div, softmax
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2


class PoolingLayer(Layer):

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):

        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx

        loss = tf.nn.sampled_softmax_loss(weights=embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!

        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        keys = inputs[0]
        query = inputs[1]
        weight = reduce_sum(keys * query, axis=-1, keep_dims=True)
        weight = tf.pow(weight, self.pow_p)  # [x,k_max,1]

        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.log1p(tf.cast(inputs[2], dtype="float32")) / tf.log(2.)  # hist_len
                )
            ), dtype="int64")
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)

        weight = softmax(weight, dim=1, name="weight")
        output = reduce_sum(keys * weight, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size)

    def get_config(self, ):
        config = {'k_max': self.k_max, 'pow_p': self.pow_p}
        base_config = super(LabelAwareAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Similarity(Layer):

    def __init__(self, gamma=1, axis=-1, type='cos', **kwargs):
        self.gamma = gamma
        self.axis = axis
        self.type = type
        super(Similarity, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Similarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        query, candidate = inputs
        if self.type == "cos":
            query_norm = tf.norm(query, axis=self.axis)
            candidate_norm = tf.norm(candidate, axis=self.axis)
        cosine_score = reduce_sum(tf.multiply(query, candidate), -1)
        if self.type == "cos":
            cosine_score = div(cosine_score, query_norm * candidate_norm + 1e-8)
        cosine_score = tf.clip_by_value(cosine_score, -1, 1.0) * self.gamma
        return cosine_score

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'gamma': self.gamma, 'axis': self.axis, 'type': self.type}
        base_config = super(Similarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        behavior_embddings, seq_len = inputs
        batch_size = tf.shape(behavior_embddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])

        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad)
            weight = tf.nn.softmax(routing_logits_with_padding)
            behavior_embdding_mapping = tf.tensordot(behavior_embddings, self.bilinear_mapping_matrix, axes=1)
            Z = tf.matmul(weight, behavior_embdding_mapping)
            interest_capsules = squash(Z)
            delta_routing_logits = reduce_sum(
                tf.matmul(interest_capsules, tf.transpose(behavior_embdding_mapping, perm=[0, 2, 1])),
                axis=0, keep_dims=True
            )
            self.routing_logits.assign_add(delta_routing_logits)
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])
        return interest_capsules

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squash(inputs):
    vec_squared_norm = reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed


class EmbeddingIndex(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
