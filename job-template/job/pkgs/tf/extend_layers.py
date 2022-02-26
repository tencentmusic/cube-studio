
from .feature_util import *
from .helperfuncs import create_activate_func, create_regularizer, squash


class VocabLookupLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size=None, vocab_list=None, vocab_file=None, key_dtype=None, hash_type='hash',
                 num_oov_buckets=None,
                 name="vocab_lookup_layer", trainable=False, **kwargs):
        super(VocabLookupLayer, self).__init__(name=name, trainable=trainable, **kwargs)

        assert (vocab_size is not None and vocab_size > 0 and not vocab_list and not vocab_file) or \
               ((vocab_size is None or vocab_size <= 0) and (isinstance(vocab_list, (tuple, list)) and vocab_list)
                and not vocab_file) or \
               ((vocab_size is None or vocab_size <= 0) and not vocab_list and isinstance(vocab_file, str)
                and vocab_file.strip()), \
            "one and only one of vocab_size, vocab_list, vocab_file must be specified, got {}, ({}){}..., {}" \
                .format(vocab_size, len(vocab_list), vocab_list[:10], vocab_file)

        self.num_oov_buckets = num_oov_buckets

        if isinstance(key_dtype, str):
            from job.pkgs.tf.feature_util import dtype_from_str
            key_dtype = dtype_from_str(key_dtype)

        if isinstance(vocab_list, (tuple, list)) and vocab_list:
            self.vocab_list = sorted(vocab_list)
            self.oov_tag = len(vocab_list)
            self.hash_type = None
        elif isinstance(vocab_file, str) and vocab_file:
            if not os.path.isfile(vocab_file):
                raise RuntimeError("vocab_file '{}' not exist".format(vocab_file))
            vocab_set = set()
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line in vocab_set:
                        raise RuntimeError("word '{}' duplicated in vocab file '{}'".format(vocab_file))
                    vocab_set.add(line)
            if not vocab_set:
                raise RuntimeError("vocab file '{}' is empty".format(vocab_file))
            self.vocab_list = sorted(vocab_set)
            self.oov_tag = len(self.vocab_list)
            self.hash_type = None
        elif vocab_size == 1:
            self.hash_type = 'fix'
            print("{}: vocab_size=1, set hash_type='fix'".format(self.name))
            self.oov_tag = vocab_size
        else:
            if isinstance(hash_type, str) and hash_type.strip():
                hash_type = hash_type.strip().lower()
                if hash_type not in ['hash', 'mod']:
                    raise RuntimeError("{}: 'hash_type' should be 'hash'/'mod', got '{}'"
                                       .format(self.name, hash_type))
                if hash_type == 'mod':
                    if not key_dtype.is_integer and not key_dtype.is_floating:
                        raise RuntimeError("{}: 'mod' hash type only support int/float, got key_dtype={}"
                                           .format(self.name, key_dtype))
                self.hash_type = hash_type
            else:
                self.hash_type = 'hash'
            self.oov_tag = vocab_size

        if not self.hash_type:
            if self.vocab_list == list(range(self.oov_tag)):
                print("{}: vocab list {} is range(0, {}), will left inputs unchanged"
                      .format(self.name, self.vocab_list, self.oov_tag))
                self.vocab_lookup_table = None
            else:
                words = []
                indices = []
                for i, w in enumerate(self.vocab_list):
                    words.append(w)
                    indices.append(i)
                if self.num_oov_buckets:
                    self.vocab_lookup_table = tf.lookup.StaticVocabularyTable(
                        tf.lookup.KeyValueTensorInitializer(words, indices, key_dtype=key_dtype, value_dtype=tf.int64),
                        self.num_oov_buckets, name=self.name + "/lookup_table")
                else:
                    if key_dtype.is_integer:
                        value_dtype = key_dtype
                    else:
                        value_dtype = tf.int64
                    self.vocab_lookup_table = tf.lookup.StaticHashTable(
                        tf.lookup.KeyValueTensorInitializer(words, indices, key_dtype=key_dtype,
                                                            value_dtype=value_dtype),
                        self.oov_tag, name=self.name + "/lookup_table")
                print("{}: created lookup table".format(self.name))
        else:
            print("{}: will use hash '{}'".format(self.name, self.hash_type))
        self.key_dtype = key_dtype

    @property
    def vocab_key_dtype(self):
        return self.key_dtype

    @property
    def vocab_value_dtype(self):
        if self.hash_type == 'hash':
            return tf.int64
        elif self.hash_type in ['mod', 'fix']:
            return self.key_dtype
        return self.vocab_lookup_table.value_dtype

    @tf.function
    def call(self, inputs, **kwargs):
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        is_sparse = isinstance(inputs, tf.SparseTensor)
        if self.hash_type == 'fix':
            if is_sparse:
                return tf.SparseTensor(inputs.indices, tf.zeros_like(inputs.values), inputs.dense_shape)
            return tf.zeros_like(inputs)
        if self.hash_type:
            # tf.strings.* api支持RaggedTensor，不支持SparseTensor
            if is_sparse:
                if self.hash_type == 'hash':
                    if inputs.dtype != tf.string:
                        str_vals = tf.strings.as_string(inputs.values, name=self.name + '_sparse_as_string')
                    else:
                        str_vals = inputs.values
                    val_hashs = tf.strings.to_hash_bucket_fast(str_vals, self.oov_tag,
                                                               name=self.name + '_sparse_to_hash_bucket')
                else:
                    val_hashs = tf.math.mod(inputs.values, self.oov_tag,
                                            name=self.name + '_sparse_mod_{}'.format(self.oov_tag))
                return tf.SparseTensor(inputs.indices, val_hashs, inputs.dense_shape)

            if self.hash_type == 'hash':
                if inputs.dtype != tf.string:
                    inputs = tf.strings.as_string(inputs, name=self.name + "_as_string")
                return tf.strings.to_hash_bucket_fast(inputs, self.oov_tag, name=self.name + "_to_hash_bucket")
            return tf.math.mod(inputs, self.oov_tag, name=self.name + '_mod_{}'.format(self.oov_tag))
        else:
            if self.vocab_lookup_table is None:
                return inputs
            # lookup table支持SparseTensor，不支持RaggedTensor
            if is_ragged:
                rag_lengths = inputs.nested_row_lengths()
                ids = self.vocab_lookup_table.lookup(inputs.to_tensor(), name=self.name + 'ragged_table_lookup')
                return tf.RaggedTensor.from_tensor(ids, rag_lengths)
            return self.vocab_lookup_table.lookup(inputs, name=self.name + '_table_lookup')

    def get_vocab_size(self):
        return self.oov_tag if not self.num_oov_buckets else self.oov_tag + self.num_oov_buckets

    def get_config(self):
        config = super(VocabLookupLayer, self).get_config()
        if self.hash_type:
            config.update({
                'vocab_size': self.oov_tag,
                'key_dtype': self.key_dtype.name,
                'hash_type': self.hash_type,
                'num_oov_buckets': self.num_oov_buckets
            })
        else:
            config.update({
                'vocab_list': list(self.vocab_list),
                'key_dtype': self.key_dtype.name,
                'num_oov_buckets': self.num_oov_buckets
            })
        return config

    @classmethod
    def from_config(cls, config):
        from job.pkgs.tf.feature_util import dtype_from_str
        key_dtype_str = config.get('key_dtype')
        config['key_dtype'] = dtype_from_str(key_dtype_str)
        return cls(**config)


class PoolingEmbeddingLayer(tf.keras.layers.Layer):
    """
    PoolingEmbeddingLayer此后只负责进行VocabLookup之后的Embedding工作
    关于Embedding的后处理如max pooling, mean pooling等操作将交给其他函数或Layer处理
    """
    def __init__(self, vocab_size, embedding_dim, embedding_initializer=None, embedding_l2_reg=None,
                 embedding_l1_reg=None, embedding_combiner=None, name="pooling_embedding_layer",
                 trainable=True, max_len=None, padding=None, **kwargs):
        super(PoolingEmbeddingLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        assert vocab_size > 0, \
            "vocab_size must be a positive integer, got '{}': {}".format(type(vocab_size), vocab_size)
        assert embedding_dim > 0, \
            "embedding_dim must be a positive integer, got '{}': {}".format(type(embedding_dim), embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_initializer = embedding_initializer
        self.embedding_l2_reg = embedding_l2_reg
        self.embedding_l1_reg = embedding_l1_reg
        self.embedding_combiner = embedding_combiner.strip().lower() if embedding_combiner else None
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size + 2, self.embedding_dim,
                                                         self.embedding_initializer or 'uniform',
                                                         create_regularizer(embedding_l1_reg, embedding_l2_reg),
                                                         name=self.name + "/inner_embedding_layer")
        self.padding_value = padding or vocab_size + 1
        self.max_len = max_len or 128

    def build(self, input_shape):
        self.embedding_layer.build(input_shape)
        super(PoolingEmbeddingLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        is_sparse = isinstance(inputs, tf.SparseTensor)
        is_ragged = isinstance(inputs, tf.RaggedTensor)

        if is_sparse:
            # [batch, n, seq_len]
            inputs = tf.sparse.to_dense(inputs, name=self.name + "_sparse_to_dense")

        # 如果输入是不定长的序列向量且embedding_combiner带有seq, 则将raggedTensor转换为最大长度为max_len, padding的值为设置的padding_value的Tensor
        if is_ragged and (self.embedding_combiner == 'seq_pad' or self.embedding_combiner == 'seq_nopad'):
            if self.embedding_combiner == 'seq_pad':
                inputs = inputs.to_tensor(default_value=self.padding_value, shape=[None, None, self.max_len])
                # inputs = tf.pad(inputs, [[0,0], [0,0] [0, self.max_len-inputs.shape[-1]]], constant_values=self.padding_value) # [batch, 1, max_len]
            else:
                inputs = inputs.to_tensor(default_value=self.padding_value)  # [batch, 1, len]
            is_ragged = False

        # if ragged: [batch, n, max_len, dim] else [batch, n, seq_len, dim]
        embeddings = self.embedding_layer(inputs)
        weights = kwargs.get("weights")
        if weights is not None:
            if isinstance(weights, tf.SparseTensor):
                # [batch, n, seq_len]
                weights = tf.sparse.to_dense(weights, name=self.name + "_weights_sparse_to_dense")
            elif isinstance(weights, tf.RaggedTensor):
                # [batch, n, seq_len]
                weights = weights.to_tensor(name=self.name + "_weights_ragged_to_dense")
            else:
                # [batch, n, 1]
                weights = tf.expand_dims(weights, axis=-1)

            if is_ragged:
                # [batch, n, seq_len, dim]
                embeddings = embeddings.to_tensor()

            # [batch, n, seq_len, 1]
            weights = tf.expand_dims(weights, axis=-1)
            # [batch, n, seq_len, dim]
            if weights.dtype != embeddings.dtype:
                weights = tf.cast(weights, dtype=embeddings.dtype,
                                  name=self.name + '_cast_weight_' + embeddings.dtype.name)
            embeddings = embeddings * weights

        if embeddings.dtype != self._dtype_policy.compute_dtype:
            embeddings = tf.cast(embeddings, self._dtype_policy.compute_dtype,
                                 name=self.name + '_mp_cast2' + self._dtype_policy.compute_dtype)

        return embeddings, weights

    def get_embedding_matrix(self):
        return self.embedding_layer.embeddings

    def compute_mask_histlen(self, inputs, return_seqlist=False):
        # 计算输入向量的mask和序列真实长度, 默认会返回序列mask, 如果设置return_seqlist为True, 则会返回序列真实长度交给外部模型去处理
        # 如果是raggedTensor则可以直接计算得到, 如果不是raggedTensor, 则需要判断其使用padding_value填充的数量和位置来确定序列真实长度
        # [batch, len]
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        if is_ragged:
            row_lengths = inputs.row_lengths(-1).to_tensor()
            if return_seqlist:
                return tf.reshape(row_lengths, [-1, 1])
            else:
                return tf.sequence_mask(row_lengths, self.max_len)
        else:
            pad_inputs = tf.pad(inputs, [[0, 0], [0, 1]], constant_values=self.padding_value)
            row_lengths = tf.argmax(tf.cast(tf.equal(pad_inputs, self.padding_value), tf.int32), axis=1)
            if return_seqlist:
                # pad_inputs = tf.pad(inputs, [[0,0],[0,1]], constant_values=self.padding_value)
                # row_lengths = tf.argmax(tf.cast(tf.equal(pad_inputs, self.padding_value), tf.int32), axis=1)
                return tf.reshape(row_lengths, [-1, 1])
            else:
                return tf.sequence_mask(row_lengths, self.max_len)
                # return tf.where(tf.equal(inputs, self.padding_value), False, tf.fill(inputs.shape, True))

    def get_config(self):
        config = super(PoolingEmbeddingLayer, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'embedding_initializer': self.embedding_initializer,
            'embedding_l2_reg': self.embedding_l2_reg,
            'embedding_l1_reg': self.embedding_l1_reg,
            'embedding_combiner': self.embedding_combiner
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VocabEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, vocab_size=None, vocab_list=None, vocab_file=None, vocab_dtype=None,
                 vocab_hash_type='hash', num_oov_buckets=None, embedding_initializer=None, embedding_l2_reg=None,
                 embedding_l1_reg=None,
                 embedding_combiner=None, name="vocab_embedding_layer", trainable=True, max_len=None, padding=None,
                 **kwargs):
        super(VocabEmbeddingLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.vocab_lookup_layer = VocabLookupLayer(vocab_size, vocab_list, vocab_file, vocab_dtype, vocab_hash_type,
                                                   num_oov_buckets,
                                                   name=self.name + "/vocab_lookup_layer")
        self.embedding_layer = PoolingEmbeddingLayer(self.vocab_lookup_layer.get_vocab_size() + 1,
                                                     embedding_dim, embedding_initializer, embedding_l2_reg,
                                                     embedding_l1_reg, embedding_combiner,
                                                     name=self.name + "/pooling_embedding_layer",
                                                     max_len=max_len, padding=padding)
        self._output_shape = None
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.vocab_list = vocab_list
        self.vocab_file = vocab_file
        self.vocab_dtype = vocab_dtype
        self.vocab_hash_type = vocab_hash_type
        self.embedding_initializer = embedding_initializer
        self.embedding_l2_reg = embedding_l2_reg
        self.embedding_l1_reg = embedding_l1_reg
        self.embedding_combiner = embedding_combiner

    def build(self, input_shape):
        self.embedding_layer.build(input_shape)
        super(VocabEmbeddingLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        is_ragged = isinstance(inputs, tf.RaggedTensor)  # kwargs.get('is_ragged')
        is_sparse = isinstance(inputs, tf.SparseTensor)  # kwargs.get('is_sparse')
        # if is_ragged: [batch, n, None], if is_sparse: [batch, n, seq_len], else [batch, n]
        indices = self.vocab_lookup_layer(inputs)
        if not is_ragged and not is_sparse:
            # [batch, n, 1]
            indices = tf.expand_dims(indices, -1)
        # [batch, n, dim]
        embeddings, weights = self.embedding_layer(indices, **kwargs)
        return embeddings, weights # 需要注意此处VocabEmbeddingLayer的输出包含了两部分, 且embeddings没有经过pooling等后处理操作

    def compute_mask_histlen(self, inputs, return_seqlist=False):
        indices = self.vocab_lookup_layer(inputs)
        return self.embedding_layer.compute_mask_histlen(indices, return_seqlist)

    def get_embedding_matrix(self):
        return self.embedding_layer.get_embedding_matrix()

    def get_vocab_size(self):
        return self.vocab_lookup_layer.get_vocab_size()

    def get_vocab_key_dtype(self):
        return self.vocab_lookup_layer.vocab_key_dtype

    def get_vocab_value_dtype(self):
        return self.vocab_lookup_layer.vocab_value_dtype

    def get_index_dtype(self):
        return self.vocab_lookup_layer

    def word_to_index(self, words):
        return self.vocab_lookup_layer(words)

    def get_config(self):
        config = super(VocabEmbeddingLayer, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
            'vocab_list': self.vocab_list,
            'vocab_file': self.vocab_file,
            'vocab_dtype': self.vocab_dtype.name,
            'vocab_hash_type': self.vocab_hash_type,
            'embedding_initializer': self.embedding_initializer,
            'embedding_l2_reg': self.embedding_l2_reg,
            'embedding_l1_reg': self.embedding_l1_reg,
            'embedding_combiner': self.embedding_combiner
        })
        return config

    @classmethod
    def from_config(cls, config):
        from job.pkgs.tf.feature_util import dtype_from_str
        dtype_str = config.get('vocab_dtype')
        config['vocab_dtype'] = dtype_from_str(dtype_str)
        return cls(**config)


class VocabMultiHotLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size=None, vocab_list=None, vocab_file=None, vocab_dtype=None,
                 vocab_hash_type='hash', num_oov_buckets=None, name="vocab_onehot_layer", **kwargs):
        super(VocabMultiHotLayer, self).__init__(name=name, trainable=False, **kwargs)
        self.vocab_lookup_layer = VocabLookupLayer(vocab_size, vocab_list, vocab_file, vocab_dtype, vocab_hash_type,
                                                   num_oov_buckets,
                                                   self.name + "/vocab_lookup_layer")
        self.vocab_size = vocab_size
        self.vocab_list = vocab_list
        self.vocab_file = vocab_file
        self.vocab_dtype = vocab_dtype
        self.vocab_hash_type = vocab_hash_type

    @tf.function
    def call(self, inputs, **kwargs):
        is_ragged = isinstance(inputs, tf.RaggedTensor)
        is_sparse = isinstance(inputs, tf.SparseTensor)
        # if is_ragged: [batch, n, None], if is_sparse: [batch, n, seq_len], else [batch, n]
        indices = self.vocab_lookup_layer(inputs)
        if not indices.dtype.is_integer:
            indices = tf.cast(indices, tf.int32, name=self.name + "_cast_indices_to_int")

        if not is_ragged and not is_sparse:
            # [batch, n, 1]
            indices = tf.expand_dims(indices, -1)

        if is_sparse:
            # [batch, n, seq_len]
            indices = tf.sparse.to_dense(indices, default_value=-1, name=self.name + "_sparse_to_dense")

        vocab_size = self.vocab_lookup_layer.get_vocab_size()
        # if is_ragged: [batch, n, None, vocab_size], if is_sparse: [batch, n, seq_len, vocab_size]
        # else: [batch, n, 1, vocab_size]
        onehot = tf.one_hot(indices, vocab_size, dtype=self._dtype_policy.compute_dtype,
                            name=self.name + "_onehot_{}".format(vocab_size))

        weights = kwargs.get("weights")
        if weights is not None:
            if isinstance(weights, tf.SparseTensor):
                # [batch, n, seq_len]
                weights = tf.sparse.to_dense(weights, name=self.name + "_weights_sparse_to_dense")
            elif isinstance(weights, tf.RaggedTensor):
                # [batch, n, seq_len]
                weights = weights.to_tensor(name=self.name + "_weights_ragged_to_dense")

            if is_ragged:
                # [batch, n, seq_len, vocab_size]
                onehot = onehot.to_tensor()

            # [batch, n, seq_len, 1]
            weights = tf.expand_dims(weights, axis=-1)
            # [batch, n, seq_len, vocab_size]
            onehot = onehot * weights
        # [batch, n, vocab_size]
        onehot = tf.reduce_mean(onehot, axis=-2, name=self.name + '_pooling_onehot')
        if onehot.dtype != self._dtype_policy.compute_dtype:
            onehot = tf.cast(onehot, self._dtype_policy.compute_dtype,
                             name=self.name + '_mp_cast2' + self._dtype_policy.compute_dtype)

        if is_ragged and weights is None:
            return onehot.to_tensor()
        return onehot

    def get_config(self):
        config = super(VocabMultiHotLayer, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'vocab_list': self.vocab_list,
            'vocab_file': self.vocab_file,
            'vocab_dtype': self.vocab_dtype.name,
            'vocab_hash_type': self.vocab_hash_type
        })
        return config

    @classmethod
    def from_config(cls, config):
        from job.pkgs.tf.feature_util import dtype_from_str
        dtype_str = config.get('vocab_dtype')
        config['vocab_dtype'] = dtype_from_str(dtype_str)
        return cls(**config)


class DNNLayer(tf.keras.layers.Layer):
    def __init__(self, layer_widthes, hidden_active_fn=None, output_active_fn=None, dropout_prob=None,
                 use_bn=False, l1_reg=None, l2_reg=None, use_bias=True, name="dnn_layer", trainable=True,
                 **kwargs):
        super(DNNLayer, self).__init__(name=name, trainable=trainable, **kwargs)

        self.dense_layers = []
        self.bn_layers = None
        self.dropout_layers = None
        self.hidden_active_layers = None
        self.output_active_layer = None

        self.layer_widthes = layer_widthes
        self.hidden_active_fn = hidden_active_fn
        self.output_active_fn = output_active_fn.strip() if isinstance(output_active_fn, str) else None
        self.dropout_prob = dropout_prob
        self.use_bn = use_bn
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_bias = use_bias

        if self.use_bn:
            self.bn_layers = []
        if self.dropout_prob:
            self.dropout_layers = []
        if self.hidden_active_fn:
            self.hidden_active_layers = []
        if self.output_active_fn:
            if output_active_fn.lower() == 'dice':
                self.output_active_layer = create_activate_func(output_active_fn)(
                    name=self.name + "/output_act_%s" % self.output_active_fn)
            elif output_active_fn.lower() == 'prelu':
                self.output_active_layer = create_activate_func(output_active_fn)(
                    name=self.name + "/output_act_%s" % self.output_active_fn)
            else:
                self.output_active_layer = tf.keras.layers.Activation(
                    create_activate_func(self.output_active_fn),
                    name=self.name + "/output_act_%s" % self.output_active_fn)
        for i, layer_width in enumerate(self.layer_widthes):
            self.dense_layers.append(tf.keras.layers.Dense(
                layer_width,
                use_bias=use_bias,
                kernel_regularizer=create_regularizer(self.l1_reg, self.l2_reg),
                name=self.name + "/hidden_layer_%s" % i))
            if self.hidden_active_fn and i < len(self.layer_widthes) - 1:
                if isinstance(self.hidden_active_fn, str):
                    act = self.hidden_active_fn.strip()
                elif isinstance(self.hidden_active_fn, (tuple, list)):
                    if i < len(self.hidden_active_fn) and isinstance(self.hidden_active_fn[i], str):
                        act = self.hidden_active_fn[i].strip()
                    else:
                        act = None
                if act:
                    if act.lower() == 'dice':
                        self.hidden_active_layers.append(
                            create_activate_func(act)(name=self.name + "/hidden_act_%s_%s" % (act, i)))
                    elif act.lower() == 'prelu':
                        self.hidden_active_layers.append(
                            create_activate_func(act)(name=self.name + "/hidden_act_%s_%s" % (act, i)))
                    else:
                        self.hidden_active_layers.append(tf.keras.layers.Activation(
                            create_activate_func(act),
                            name=self.name + "/hidden_act_%s_%s" % (act, i)))
                else:
                    self.hidden_active_layers.append(None)
            if self.use_bn:
                if isinstance(use_bn, (tuple, list)):
                    if i < len(self.use_bn) and self.use_bn[i]:
                        self.bn_layers.append(tf.keras.layers.BatchNormalization(name=self.name + "/bn_layer_%s" % i))
                    else:
                        self.bn_layers.append(None)
                else:
                    self.bn_layers.append(tf.keras.layers.BatchNormalization(name=self.name + "/bn_layer_%s" % i))
            if self.dropout_prob and i < len(self.layer_widthes) - 1:
                if isinstance(self.dropout_prob, (tuple, list)):
                    if i < len(self.dropout_prob):
                        dropout = self.dropout_prob[i]
                    else:
                        dropout = None
                else:
                    dropout = self.dropout_prob
                if isinstance(dropout, float) and 0 < dropout < 1:
                    self.dropout_layers.append(tf.keras.layers.Dropout(dropout,
                                                                       name=self.name + "/dropout_layer_%s_%s" %
                                                                            (dropout, i)))
                else:
                    self.dropout_layers.append(None)

        print("DNNLayer '{}': output_active_layer={}, hidden_active_layers={}, bn_layers={}, dropout_layers={}"
              .format(self.name, self.output_active_layer, self.hidden_active_layers, self.bn_layers,
                      self.dropout_layers))

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        y = inputs
        for i in range(len(self.layer_widthes)):
            y = self.dense_layers[i](y)
            if self.use_bn:
                bn_layer = self.bn_layers[i]
                if bn_layer is not None:
                    y = bn_layer(y, training=training)
            if self.hidden_active_fn and i < len(self.layer_widthes) - 1:
                hidden_act_layer = self.hidden_active_layers[i]
                if hidden_act_layer is not None:
                    y = hidden_act_layer(y)
            elif i == len(self.layer_widthes) - 1 and self.output_active_fn:
                y = self.output_active_layer(y)
            if self.dropout_prob and i < len(self.layer_widthes) - 1:
                dropout_layer = self.dropout_layers[i]
                if dropout_layer is not None:
                    y = dropout_layer(y, training=training)

        return y

    def get_config(self):
        config = super(DNNLayer, self).get_config()
        config.update({
            'layer_widthes': self.layer_widthes,
            'hidden_active_fn': self.hidden_active_fn,
            'output_active_fn': self.output_active_fn,
            'dropout_prob': self.dropout_prob,
            'use_bn': self.use_bn,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'use_bias': self.use_bias
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BucktizeLayer(tf.keras.layers.Layer):
    def __init__(self, boundaries, name="buckitze_layer", trainable=False, **kwargs):
        super(BucktizeLayer, self).__init__(name=name, trainable=False, **kwargs)
        self.boundaries = sorted(boundaries)

    @tf.function
    def call(self, inputs, **kwargs):
        return tf.raw_ops.Bucketize(input=inputs, boundaries=self.boundaries)

    def get_config(self):
        config = super(BucktizeLayer, self).get_config()
        config.update({
            'boundaries': self.boundaries
        })
        return config


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, field_vocab_specs, with_logits=False, use_bias=True, embedding_l1_reg=None,
                 embedding_l2_reg=None, name="fm_layer", trainable=True, **kwargs):
        super(FMLayer, self).__init__(name=name, trainable=trainable, **kwargs)

        assert isinstance(embedding_dim, int) and embedding_dim > 0, \
            "'embedding_dim' should be a positive int, got '{}': {}" \
                .format(type(embedding_dim), embedding_dim)

        assert isinstance(field_vocab_specs, dict) and field_vocab_specs, \
            "'field_vocab_specs' should be a non-empty dict, got '{}': {}" \
                .format(type(field_vocab_specs), field_vocab_specs)

        if not all([isinstance(s, dict) for s in field_vocab_specs.values()]):
            raise RuntimeError("items of 'field_vocab_specs' should all be dict, got: {}"
                               .format(field_vocab_specs))

        # self.field_weights = {}
        self.categorical_fields_weights = {}
        self.categorical_fields_latent_matrixs = {}
        self.field_is_dense = {}

        weight_emb_layers = {}
        latent_matrix_emb_layers = {}
        for name, vocab_spec in field_vocab_specs.items():
            spec = vocab_spec.copy()
            spec.pop('embedding_dim', None)
            spec.pop('weight_col', None)
            is_dense = spec.pop('is_dense')
            self.field_is_dense[name] = is_dense
            if is_dense:
                continue
            embedding_name = spec.pop('embedding_name', '').strip()
            if embedding_name and embedding_name in weight_emb_layers:
                weight_emb_layer = weight_emb_layers.get(embedding_name)
                latent_matrix_emb_layer = latent_matrix_emb_layers.get(embedding_name)
            else:
                weight_emb_layer = VocabEmbeddingLayer(1, embedding_l1_reg=embedding_l1_reg,
                                                       embedding_l2_reg=embedding_l2_reg,
                                                       name=self.name + "/" + (embedding_name or name) + '_weights',
                                                       **spec)
                latent_matrix_emb_layer = VocabEmbeddingLayer(embedding_dim, embedding_l1_reg=embedding_l1_reg,
                                                              embedding_l2_reg=embedding_l2_reg,
                                                              name=self.name + "/" + (embedding_name or name) +
                                                                   '_latent_matrixes',
                                                              **spec)
                if embedding_name:
                    weight_emb_layers[embedding_name] = weight_emb_layer
                    latent_matrix_emb_layers[embedding_name] = latent_matrix_emb_layer

            self.categorical_fields_weights[name] = PoolingProcessLayer(
                weight_emb_layer, name=weight_emb_layer.name + '_pooling_processor')
            self.categorical_fields_latent_matrixs[name] = PoolingProcessLayer(
                latent_matrix_emb_layer, name=latent_matrix_emb_layer.name + '_pooling_processor')

        dense_cnt = len(field_vocab_specs) - len(self.categorical_fields_latent_matrixs)
        print("{} fields, dense {}, categorical {}".format(len(field_vocab_specs), dense_cnt,
                                                           len(self.categorical_fields_latent_matrixs)))
        if dense_cnt > 0:
            self.dense_fields_weight = self.add_weight("dense_fields_weight", (dense_cnt, 1),
                                                       initializer=tf.keras.initializers.Zeros(),
                                                       trainable=True)
            self.dense_fields_latent_matrix = self.add_weight("dense_fields_latent_matrix",
                                                              (dense_cnt, embedding_dim),
                                                              initializer=tf.keras.initializers.GlorotNormal(),
                                                              trainable=True)

        if use_bias:
            self.bias = self.add_weight("bias", (1,), initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.embedding_dim = embedding_dim
        self.field_vocab_specs = field_vocab_specs
        self.with_logits = with_logits
        self.use_bias = use_bias
        self.embedding_l1_reg = embedding_l1_reg
        self.embedding_l2_reg = embedding_l2_reg

    @tf.function
    def call(self, inputs, **kwargs):
        dense_vals = []
        categorical_weights = []
        categorical_latents = []
        any_in = list(inputs.values())[0]
        if isinstance(any_in, tf.RaggedTensor):
            batch_size = any_in.nrows()
        else:
            batch_size = tf.shape(any_in)[0]
        for name, is_dense in self.field_is_dense.items():
            # [batch, 1]
            val = inputs[name]
            if is_dense:
                if val.dtype != self._dtype_policy.compute_dtype:
                    val = tf.cast(val, self._dtype_policy.compute_dtype,
                                  name=name + "_mp_cast2" + self._dtype_policy.compute_dtype)
                # [batch, 1, 1]
                val = tf.reshape(val, [batch_size, 1, -1], name=self.name + '_' + name + '_dense_val_reshape')
                dense_vals.append(val)
            else:
                field_spec = self.field_vocab_specs[name]
                if field_spec.get('weight_col'):
                    weights = inputs[field_spec['weight_col']]
                else:
                    weights = None
                w = self.categorical_fields_weights[name](val, weights=weights)
                m = self.categorical_fields_latent_matrixs[name](val, weights=weights)
                # [batch, 1, 1]
                w = tf.reshape(w, [batch_size, -1, 1], name=self.name + '_' + name + '_w_reshape')
                # [batch, 1, embedding_dim]
                m = tf.reshape(m, [batch_size, -1, self.embedding_dim], name=self.name + '_' + name + '_m_reshape')
                categorical_weights.append(w)
                categorical_latents.append(m)

        if len(dense_vals):
            # [batch, #dense_fields, 1]
            dense_val_concat = tf.concat(dense_vals, axis=1, name=self.name + '_dense_val_concat')
            if self.dense_fields_weight.dtype != self._dtype_policy.compute_dtype:
                dense_fields_weight = tf.cast(self.dense_fields_weight, self._dtype_policy.compute_dtype,
                                              name="dense_fields_weight_cast2" + self._dtype_policy.compute_dtype)
            else:
                dense_fields_weight = self.dense_fields_weight
            # [batch, #dense_fields, 1] * [#dense_fields, 1] = [batch, #dense_fields, 1]
            dense_linear_vals = tf.multiply(dense_val_concat, dense_fields_weight,
                                            name=self.name + '_dense_val_X_weights')

            if self.dense_fields_latent_matrix.dtype != self._dtype_policy.compute_dtype:
                dense_fields_latent_matrix = tf.cast(self.dense_fields_latent_matrix, self._dtype_policy.compute_dtype,
                                                     name="dense_fields_latent_matrix_cast2" +
                                                          self._dtype_policy.compute_dtype)
            else:
                dense_fields_latent_matrix = self.dense_fields_latent_matrix
            # [batch, #dense_fields, 1] * [#dense_fields, embedding_dim] = [batch, #dense_fields, embedding_dim]
            dense_latent_vals = tf.multiply(dense_val_concat, dense_fields_latent_matrix,
                                            name=self.name + '_dense_val_X_latent_matrix')

        if len(dense_vals) > 0 and len(categorical_weights) > 0:
            # [batch, #fields, 1]
            ws_concat = tf.concat(categorical_weights + [dense_linear_vals], axis=1, name=self.name + '_ws_concat')
            # [batch,  #fields, embedding_dim]
            latents_concat = tf.concat(categorical_latents + [dense_latent_vals], axis=1,
                                       name=self.name + "_latents_concat")
        elif len(dense_vals) > 0:
            # [batch, #dense_fields, 1]
            ws_concat = dense_linear_vals
            # [batch, #dense_fields, embedding_dim]
            latents_concat = dense_latent_vals
            print("{}: only contain dense fields".format(self.name))
        else:
            # [batch, #categorical_fields, 1]
            ws_concat = tf.concat(categorical_weights, axis=1, name=self.name + '_ws_concat')
            # [batch, #categorical_fields, embedding_dim]
            latents_concat = tf.concat(categorical_latents, axis=1, name=self.name + "_latents_concat")
            print("{}: only contain categorical fields".format(self.name))

        # [batch, #fields, embedding_dim]
        latents_concat = tf.reshape(latents_concat, (-1, len(self.field_is_dense), self.embedding_dim),
                                    name=self.name + '_' + name + '_latents_concat_reshape')
        # [batch, 1]
        first_order = tf.reduce_sum(ws_concat, axis=1, keepdims=False, name="1st_order_sum")

        # [batch, embedding_dim]
        per_field_emb_sum = tf.reduce_sum(latents_concat, axis=1, keepdims=False, name="per_field_emb_sum")
        # [batch, embedding_dim]
        per_field_emb_square_of_sum = tf.square(per_field_emb_sum, "per_field_emb_square_of_sum")
        # [batch, 1]
        square_of_sum = tf.reduce_sum(per_field_emb_square_of_sum, axis=1, keepdims=True, name="square_of_sum")

        # [batch, #fields, embedding_dim]
        per_field_emb_square = tf.square(latents_concat, name="per_field_emb_square")
        # [batch, embedding_dim]
        per_field_emb_sum_of_square = tf.reduce_sum(per_field_emb_square, axis=1, keepdims=False,
                                                    name="per_field_emb_sum_of_square")
        # [batch, 1]
        sum_of_square = tf.reduce_sum(per_field_emb_sum_of_square, axis=1, keepdims=True, name="sum_of_square")

        # [batch, 1]
        second_order = 0.5 * (square_of_sum - sum_of_square)
        # [batch, 1]
        logits = first_order + second_order
        # # [batch, #fields, embedding_dim]
        if self.use_bias:
            logits = logits + self.bias
        if self.with_logits:
            return logits, latents_concat
        # [batch, 1], [batch, #fields, embedding_dim]
        return tf.nn.sigmoid(logits, name="out_sigmoid"), latents_concat

    def get_config(self):
        config = super(FMLayer, self).get_config()
        vocab_specs_dict = {}
        for fn, vocab_spec in self.field_vocab_specs.items():
            if 'vocab_dtype' in vocab_spec:
                spec_copy = vocab_spec.copy()
                spec_copy['vocab_dtype'] = vocab_spec['vocab_dtype'].name
            vocab_specs_dict[fn] = spec_copy
        config.update({
            'embedding_dim': self.embedding_dim,
            'field_vocab_specs': vocab_specs_dict,
            'with_logits': self.with_logits,
            'use_bias': self.use_bias,
            'embedding_l1_reg': self.embedding_l1_reg,
            'embedding_l2_reg': self.embedding_l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        from job.pkgs.tf.feature_util import dtype_from_str
        field_vocab_specs = config.get('field_vocab_specs')
        if field_vocab_specs:
            for fn, vocab_spec in field_vocab_specs.items():
                if 'vocab_dtype' in vocab_spec:
                    dtype_str = vocab_spec['vocab_dtype']
                    vocab_spec['vocab_dtype'] = dtype_from_str(dtype_str)

        return cls(**config)

    @classmethod
    def __make_fm_field_specs_from_fcs(cls, fcs: List[fc_lib.FeatureColumn]):
        fm_field_specs = {}
        for fc in fcs:
            if isinstance(fc, fc_lib.CategoricalColumn):
                vocab_size = fc.num_buckets
                vocab_dtype = tf.int64
                is_dense = False
            else:
                if isinstance(fc, (fc_lib.EmbeddingColumn, fc_lib.SharedEmbeddingColumn, fc_lib.IndicatorColumn)):
                    raise RuntimeError("'{}' embedding/indicator feature column is not supported, got '{}'"
                                       .format(fc.name, type(fc)))
                vocab_size = 1
                vocab_dtype = tf.int32
                is_dense = True
            fm_field_specs[fc.name] = {"vocab_size": vocab_size, "vocab_dtype": vocab_dtype, "is_dense": is_dense}
        print("generate fm field specs from feature columns: {}".format(fm_field_specs))
        return fm_field_specs

    @classmethod
    def __make_fm_field_specs_from_inputs(cls, input_descs: List[InputDesc]):
        fm_field_specs = {}
        for desc in input_descs:
            if desc.is_weight_col:
                continue
            if desc.val_sep or (not desc.dtype.is_floating and not desc.dtype.is_integer):
                vocab_size = desc.get_vocab_size()
                if vocab_size <= 0:
                    raise RuntimeError("input '{}' is not numberic, but not specified vocab: {}"
                                       .format(desc.name, desc))
                vocab_dtype = desc.dtype
                is_dense = False
            else:
                vocab_size = desc.get_vocab_size()
                is_dense = vocab_size <= 0
                if is_dense:
                    vocab_size = 1
                    if desc.dtype.is_floating:
                        vocab_dtype = tf.int32
                    else:
                        vocab_dtype = desc.dtype
                else:
                    if desc.dtype.is_floating:
                        if not desc.bucket_boundaries:
                            raise RuntimeError("only integer/string input can have vocab, but input '{}' is {}: {}"
                                               .format(desc.name, desc.dtype, desc))
                        else:
                            vocab_dtype = tf.int32
                    else:
                        vocab_dtype = desc.dtype
            fm_field_specs[desc.name] = {"vocab_size": vocab_size, "vocab_dtype": vocab_dtype, "is_dense": is_dense,
                                         "embedding_name": desc.embedding_name, "vocab_hash_type": desc.hash_type,
                                         "weight_col": desc.weight_col}
        print("generate fm field specs from inputs: {}".format(fm_field_specs))
        return fm_field_specs

    @classmethod
    def create_from_model_input_config(cls, model_input_config: ModelInputConfig, groups=None, **kwargs):
        if not model_input_config.all_fcs:
            if groups:
                input_specs = model_input_config.get_inputs_by_group(groups)
            else:
                input_specs = model_input_config.all_inputs
            feat_input_specs = list(filter(lambda i: not i.is_label and not i.is_sample_weight, input_specs))
            fm_specs = cls.__make_fm_field_specs_from_inputs(feat_input_specs)
        else:
            fm_specs = cls.__make_fm_field_specs_from_fcs(model_input_config.all_fcs)

        return cls(field_vocab_specs=fm_specs, **kwargs)


class MMoELayer(tf.keras.layers.Layer):
    def __init__(self, num_tasks, num_experts, expert_layers, expert_use_bias=True,
                 expert_act='relu', expert_dropout_prob=None, expert_use_bn=False, expert_l1_reg=None,
                 expert_l2_reg=None, gate_use_bias=True, gate_l1_reg=None, gate_l2_reg=None, share_gates=False,
                 name='mmoe_layer', trainable=True, **kwargs):
        super(MMoELayer, self).__init__(name=name, trainable=trainable, **kwargs)

        assert num_tasks > 0, "'num_tasks' should be a positive integer, got '{}': {}" \
            .format(type(num_tasks), num_tasks)
        assert num_experts > 0, "'num_experts' should be a positive integer, got '{}': {}" \
            .format(type(num_experts), num_experts)
        if not expert_layers or not all([l > 0 for l in expert_layers]):
            raise RuntimeError("'expert_layers' should be a list of positive integers, got '{}': {}"
                               .format(type(expert_layers), expert_layers))

        self.experts = [DNNLayer(expert_layers, expert_act, expert_act, expert_dropout_prob, expert_use_bn,
                                 expert_l1_reg, expert_l2_reg, expert_use_bias, name=self.name + '/expert_{}'.format(i))
                        for i in range(num_experts)]
        if share_gates:
            self.gates = [tf.keras.layers.Dense(
                num_experts, use_bias=gate_use_bias,
                kernel_regularizer=create_regularizer(gate_l1_reg, gate_l2_reg),
                name=self.name + '/shared_gate'
            )]
        else:
            self.gates = [tf.keras.layers.Dense(
                num_experts, use_bias=gate_use_bias,
                kernel_regularizer=create_regularizer(gate_l1_reg, gate_l2_reg),
                name=self.name + '/gate_{}'.format(i)
            ) for i in range(num_tasks)]

        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.expert_layers = expert_layers
        self.expert_use_bias = expert_use_bias
        self.expert_act = expert_act
        self.expert_dropout_prob = expert_dropout_prob
        self.expert_use_bn = expert_use_bn
        self.expert_l1_reg = expert_l1_reg
        self.expert_l2_reg = expert_l2_reg
        self.gate_use_bias = gate_use_bias
        self.gate_l1_reg = gate_l1_reg
        self.gate_l2_reg = gate_l2_reg
        self.share_gates = share_gates

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        experts_outputs = []
        task_outputs = []
        for expert in self.experts:
            # [batch, out_dim]
            eo = expert(inputs, training)
            experts_outputs.append(eo)

        # [batch, out_dim, num_experts]
        eos = tf.stack(experts_outputs, axis=-1)

        for gate in self.gates:
            # [batch, num_experts]
            go = gate(inputs)
            go = tf.nn.softmax(go, name=gate.name + '_softmax')
            # [batch, num_exports, 1]
            go = tf.expand_dims(go, axis=-1)
            # [batch, out_dim, 1]
            mix_eos = tf.matmul(eos, go, name="eos_X_" + gate.name)
            # [batch, out_dim]
            mix_eos = tf.squeeze(mix_eos, axis=-1)
            task_outputs.append(mix_eos)

        if self.share_gates:
            return tuple(task_outputs * self.num_tasks)

        return tuple(task_outputs)

    def get_config(self):
        config = super(MMoELayer, self).get_config()
        config.update({
            'num_tasks': self.num_tasks,
            'num_experts': self.num_experts,
            'expert_layers': self.expert_layers,
            'expert_use_bias': self.expert_use_bias,
            'expert_act': self.expert_act,
            'expert_dropout_prob': self.expert_dropout_prob,
            'expert_use_bn': self.expert_use_bn,
            'expert_l1_reg': self.expert_l1_reg,
            'expert_l2_reg': self.expert_l2_reg,
            'gate_use_bias': self.gate_use_bias,
            'gate_l1_reg': self.gate_l1_reg,
            'gate_l2_reg': self.gate_l2_reg,
            'share_gates': self.share_gates
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CapsuleLayer(tf.keras.layers.Layer):
    """
    胶囊网络, 使用动态路由算法, 在MIND模型和ComiRec模型中都有使用, 用于提取用户多兴趣
    """
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3,
                 init_std=1.0, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std

    def build(self, input_shape):
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                              initializer=tf.keras.initializers.RandomNormal(stddev=self.init_std),
                                              trainable=False, name="B", dtype=tf.float32)
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=tf.keras.initializers.RandomNormal(
                                                           stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        # behavior_embedding [batch_size, max_len, input_dim]
        # seq_len [batch_size, 1] ([[seq1 len], [seq2 len], ..., [seq_b len]])
        behavior_embeddings, seq_len = inputs
        batch_size = tf.shape(behavior_embeddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])  # [batch_size, k_max]

        behavior_embedding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix,
                                                  axes=1)  # [batch_size, max_len, output_dim] (low capsule)
        self.routing_logits.assign(tf.random.normal([1, self.k_max, self.max_len],
                                                    stddev=self.init_std))  # initilize routing logits every time

        # for iteration
        for i in range(self.iteration_times):
            # padding and mask
            mask = tf.sequence_mask(seq_len_tile, self.max_len)  # [batch_size, k_max, max_len]
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]),
                                                   pad)  # [batch_size, k_max, max_len]

            # calculate weights using behavior capsule/routing (routing_logits, B)
            weight = tf.nn.softmax(routing_logits_with_padding,
                                   -2)  # [batch_size, k_max, max_len] # softmax on dim k_max

            # calculate interest capsule using behavior routing weights and behavior embeddings
            ## behavior_embedding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix, axes=1) # [batch_size, max_len, output_dim] (low capsule)
            if i == self.iteration_times - 1:
                behavior_embedding_mapping_lowcapusule = behavior_embedding_mapping
            else:
                behavior_embedding_mapping_lowcapusule = tf.stop_gradient(behavior_embedding_mapping)

            ## Z = tf.matmul(weight, behavior_embedding_mapping) # [batch_size, k_max, output_dim]
            Z = tf.matmul(weight, behavior_embedding_mapping_lowcapusule)  # [batch_size, k_max, output_dim]
            interest_capsules = squash(Z)  # [batch_size, k_max, output_dim]

            # update behavior capsule/routing (routing_logits, B)
            delta_routing_logits = tf.reduce_sum(
                tf.matmul(interest_capsules,
                          tf.transpose(behavior_embedding_mapping_lowcapusule, perm=[0, 2, 1])),
                ## tf.transpose(behavior_embedding_mapping, perm=[0, 2, 1])),
                axis=0, keepdims=True
            )  # [1, k_max, max_len]
            self.routing_logits.assign_add(
                delta_routing_logits)  # = self.routing_logits + delta_routing_logits # [1, k_max, max_len]

        # get output (high capsule)
        interest_capsules = tf.reshape(interest_capsules,
                                       [-1, self.k_max, self.out_units])  # [batch_size, k_max, output_dim]
        return interest_capsules, seq_len

    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)

    def get_config(self, ):
        config = super(CapsuleLayer, self).get_config()
        config.update({
            'input_units': self.input_units,
            'out_units': self.out_units,
            'max_len': self.max_len,
            'k_max': self.k_max,
            'iteration_times': self.iteration_times,
            'init_std': self.init_std
        })

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LabelAwareAttention(tf.keras.layers.Layer):
    """
    MIND模型中使用的LabelAwareAttention, 此处并未整合到AttentionLayer中, 因为其与MIND模型本身的兴趣数量设置以及Attention计算的指数调整有关
    """
    def __init__(self, k_max, pow_p=1, **kwargs):
        super(LabelAwareAttention, self).__init__(**kwargs)
        # For MIND Model
        self.k_max = k_max
        self.pow_p = pow_p

    def build(self, input_shape):
        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)

    @tf.function
    def call(self, inputs, training=None, **kwargs):
        # keys, query, (hist_len/seq_len)
        keys = inputs[0]  # [batch_size, k_max, output_dim]
        query = inputs[1]  # [batch_size, 1, output_dim]
        weight = tf.reduce_sum(keys * query, axis=-1, keepdims=True)
        weight = tf.pow(weight, self.pow_p)

        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.math.log1p(tf.cast(inputs[2], dtype="float32")) / tf.math.log(2.)  # hist_len
                )
            ), dtype="int64")
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)
            weight = tf.where(seq_mask, weight, padding)

        weight = tf.nn.softmax(weight, axis=1, name="weight")
        output = tf.reduce_sum(keys * weight, axis=1)  # [batch_size, output_dim]

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embedding_size)

    def get_config(self, ):
        config = super(LabelAwareAttention, self).get_config()
        config.update({
            'k_max': self.k_max,
            'pow_p': self.pow_p
        })
        return config


class AttentionLayer(tf.keras.layers.Layer):
    """
    通用的Attention框架
    实例化时传入一个包含核心的注意力计算的attention
    使用时传入query, key, value和hist_len, 其用于计算的attention将会输出attention_score, 再根据score和value进行后续操作
    """
    def __init__(self, attention):
        super(AttentionLayer, self).__init__()
        self.attn = attention

    def call(self, inputs):
        '''
        query: [batch_size, dim]; 
        key: [batch_size, seq_len, dim]; 
        value: [batch_size, seq_len, dim];
        hist_len: [batch_size, 1]
        '''
        query, key, value, hist_len = inputs
        if len(query.shape) == 2:
            query = tf.tile(tf.expand_dims(query, 1), [1, tf.shape(key)[1], 1])

        attn_score = self.attn([query, key, value, hist_len])  # [batch_size, seq_len, dim]

        if len(attn_score.shape) == 2:
            attn_score = tf.expand_dims(attn_score, -1)  # [batch_size, seq_len, 1]

        if isinstance(self.attn, LocalActivationUnit):
            if self.attn.mode == 'SUM':
                attn_score = tf.transpose(attn_score, [0, 2, 1])  # [batch_size, 1, seq_len]
                outputs = tf.matmul(attn_score, value)  # [batch_size, 1, dim]
            else:
                outputs = value * attn_score  # [batch_size, seq_len, dim]
        else:
            outputs = tf.matmul(attn_score, tf.tranpose(value, [0, 2, 1]))  # [batch_size, seq_len, seq_len]

        return outputs

    def get_config(self, ):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'attention': self.attn
        })
        return config


class LocalActivationUnit(tf.keras.layers.Layer):
    """
    DIN模型中使用的LocalActivationUnit, 其提供了一种Attention计算的方式和角度
    使用时传入query, key, value和hist_len, 输出为attention score
    需要注意的是, DIN的论文中明确表示不对Attention的结果进行softmax标准化, 但在作者提供的源码中使用了softmax, 且实验也表明不使用softmax将会影响实验结果, 因此此处提供了是否使用softmax的可选项
    """
    def __init__(self, hidden_units, activations, mode='SUM', normalization=False):
        super(LocalActivationUnit, self).__init__()

        hidden_units_cpy = hidden_units.copy()
        if hidden_units_cpy[-1] != 1:
            hidden_units_cpy.append(1)
        self.dense_layer = DNNLayer(hidden_units_cpy, activations)

        self.hidden_units = hidden_units
        self.activations = activations
        self.mode = mode
        self.normalization = normalization

    def call(self, inputs):
        '''
        query: [batch_size, seq_len dim]; 
        key: [batch_size, seq_len, dim]; 
        value: [batch_size, seq_len, dim];
        hist_len: [batch_size, 1]
        '''
        query, key, value, hist_len = inputs

        info = tf.concat([query, key, query - key, query * key], axis=-1)

        # dense
        attn = self.dense_layer(info)

        seq_len_tile = tf.tile(hist_len, [1, tf.shape(attn)[-1]])  # [batch_size, dim=1]
        mask = tf.sequence_mask(seq_len_tile, tf.shape(attn)[1])  # [batch_size, dim=1, max_len]
        mask = tf.transpose(mask, [0, 2, 1])  # [batch_size, max_len, dim]
        pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [batch_size, max_len, dim=1]

        attn = tf.where(mask, attn, pad)  # [batch_size, seq_len, dim=1]

        if self.normalization:
            attn = tf.nn.softmax(attn, -2)  # [batch_size, seq_len, dim=1]

        return attn

    def get_config(self, ):
        config = super(LocalActivationUnit, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'activations': self.activations,
            'mode': self.mode,
            'normalization': self.normalization
        })
        return config


class PositionalEncodingLayer(tf.keras.layers.Layer):
    """
    用于给序列数据增加位置编码, 方便后续使用Self-Attention进行计算
    目前仅支持使用可学习的位置编码, 后续可以增加基于Sin和Cos的固定位置编码
    """
    def __init__(self, dim, max_len=2048, learnable=True, init_std=None, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.dim = dim
        self.max_len = max_len
        self.learnable = learnable
        self.init_std = init_std or 0.02

    def build(self, input_shape):
        if self.learnable:
            self.position_embedding = self.add_weight(shape=[self.max_len, self.dim],
                                                      initializer=tf.keras.initializers.TruncatedNormal(
                                                          stddev=self.init_std),
                                                      trainable=True, name="pos_enc", dtype=tf.float32)
        else:
            self.position_embedding = self.add_weight(shape=[self.max_len, self.dim],
                                                      initializer=tf.keras.initializers.TruncatedNormal(
                                                          stddev=self.init_std),
                                                      trainable=False, name="pos_enc", dtype=tf.float32)

    @tf.function
    def call(self, inputs, position_ids=None):
        # inputs [batch, seq_len, dim]
        shapes = tf.shape(inputs)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=shapes[1]), axis=0)

        position_embeds = tf.gather(params=self.position_embedding, indices=position_ids)
        position_embeds = tf.tile(input=position_embeds, multiples=(shapes[0], 1, 1))

        return position_embeds

    def get_config(self, ):
        config = super(PositionalEncodingLayer, self).get_config()
        config.update({
            'dim': self.dim,
            'max_len': self.max_len,
            'learnable': self.learnable,
            'init_std': self.init_std,
        })
        return config


def pooling_process(inputs, embeddings, weights, embedding_combiner):
    """
    Embedding之后的后处理被拆解到此方法中, 后处理本身不包含任何参数, 可以根据不同的embedding_combiner来对embeddings进行不同处理
    将Embedding和后处理解耦的原因: 不同的特征由于其具体元素的含义可能相同(如yesterday_price和today_price都是price), 因此需要共享embedding_layer
        而原本的PoolingEmbeddingLayer只能在初始化时确定embedding_combiner, 但是不同特征其需要的embedding_combiner可能不同(如有的序列特征要求保留序列, 有的则要求对序列进行pooling)
        而如果在PoolingEmbeddingLayer的call方法中传入不同embedding_combiner, 在模型保存和加载时可能存在问题, 因此选择将这部分拆解出来
    """
    is_sparse = isinstance(inputs, tf.SparseTensor)
    is_ragged = isinstance(embeddings, tf.RaggedTensor)
    if is_sparse:
        # [batch, n, seq_len]
        mask = tf.sparse.to_dense(inputs, default_value=-1, name="pooling_mask_sparse_to_dense")
        mask = (mask >= 0)
        # [batch, n, seq_len, 1]
        mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)

    if embedding_combiner == 'max' or embedding_combiner == 'seq_max':
        if is_sparse:
            # [batch, n, seq_len, dim]
            embeddings = embeddings - (1 - mask) * 1e9
        # [batch, n, dim]
        embeddings = tf.reduce_max(embeddings, axis=-2, name='pooling_maxpooling_reduce_max')
    elif embedding_combiner == 'min' or embedding_combiner == 'seq_min':
        if is_sparse:
            # [batch, n, seq_len, dim]
            embeddings = embeddings + (1 - mask) * 1e9
        # [batch, n, dim]
        embeddings = tf.reduce_min(embeddings, axis=-2, name='pooling_minpooling_reduce_min')
    elif embedding_combiner == 'sum' or embedding_combiner == 'seq_sum':
        if is_sparse:
            # [batch, n, seq_len, dim]
            embeddings = embeddings * mask
        # [batch, n, dim]
        embeddings = tf.reduce_sum(embeddings, axis=-2, name='pooling_sumpooling_reduce_sum')
    elif embedding_combiner == 'seq_pad' or embedding_combiner == 'seq_nopad':
        # [batch, n, max_len, dim]
        pass
    else:
        if is_sparse:
            # [batch, n, seq_len, dim]
            embeddings = embeddings * mask
            # [batch, n, dim]
            embeddings = tf.reduce_sum(embeddings, axis=-2, name='pooling_sparse_meanpooling_emb_reduce_sum')
            # [batch, n, seq_len]
            mask = tf.squeeze(mask, axis=-1)
            # [batch, n, 1]
            lenths = tf.reduce_sum(mask, axis=-1, keepdims=True,
                                   name='pooling_sparse_meanpooling_len_reduce_sum')
            # [batch, n, dim]
            embeddings = embeddings / (lenths + 1e-8)
        else:
            # [batch, n, dim]
            embeddings = tf.reduce_mean(embeddings, axis=-2, name='pooling_meanpooling_reduce_mean')

    if is_ragged and weights is None:
        return embeddings.to_tensor()

    return embeddings


class PoolingProcessLayer(tf.keras.layers.Layer):
    """
    如果想要导出原本的VocabEmbeddingLayer, 即包含了Embedding和Pooling过程的Layer, 可使用PoolingProcessLayer
    PoolingProcessLayer在实例化时接收一个实例化的VocabEmbeddingLayer, 并将embedding_combiner固定下来, 在计算时先得到embedding结果, 再进行embedding的后处理得到结果
    """
    def __init__(self, embedding_layer, embedding_combiner=None,
                 name="pooling_process_layer", trainable=True, **kwargs):
        super(PoolingProcessLayer, self).__init__(name=name, trainable=trainable, **kwargs)

        self.embedding_layer = embedding_layer
        self.embedding_combiner = embedding_combiner or embedding_layer.embedding_combiner
        self.embedding_dim = embedding_layer.embedding_dim
        self.vocab_size = embedding_layer.vocab_size
        self.vocab_list = embedding_layer.vocab_list
        self.vocab_file = embedding_layer.vocab_file
        self.vocab_dtype = embedding_layer.vocab_dtype
        self.vocab_hash_type = embedding_layer.vocab_hash_type
        self.embedding_initializer = embedding_layer.embedding_initializer
        self.embedding_l2_reg = embedding_layer.embedding_l2_reg
        self.embedding_l1_reg = embedding_layer.embedding_l1_reg

    @tf.function
    def call(self, inputs, **kwargs):
        embeddings, weights = self.embedding_layer(inputs, **kwargs)
        embeddings = pooling_process(inputs, embeddings, weights, self.embedding_combiner)

        return embeddings

    def compute_mask_histlen(self, inputs, return_seqlist=False):
        return self.embedding_layer.compute_mask_histlen(inputs, return_seqlist)

    def get_embedding_matrix(self):
        return self.embedding_layer.get_embedding_matrix()

    def get_vocab_size(self):
        return self.embedding_layer.get_vocab_size()

    def get_vocab_key_dtype(self):
        return self.embedding_layer.get_vocab_key_dtype()

    def get_vocab_value_dtype(self):
        return self.embedding_layer.get_vocab_value_dtype()

    def get_index_dtype(self):
        return self.embedding_layer.get_index_dtype()

    def word_to_index(self, words):
        return self.embedding_layer.word_to_index(words)


class ModelInputLayer(tf.keras.layers.Layer):
    def __init__(self, model_input_config: ModelInputConfig, groups=None, auto_create_embedding=False,
                 name="model_input_layer", trainable=True, **kwargs):
        super(ModelInputLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        if groups:
            input_descs = model_input_config.get_inputs_by_group(groups)
        else:
            input_descs = model_input_config.all_inputs
        if not input_descs:
            raise RuntimeError("{}: got no input descriptions from model input config, groups={}"
                               .format(self.name, groups))

        self.feat_input_descs = list(filter(lambda d: not d.is_label and not d.is_sample_weight and not d.exclude,
                                            input_descs))
        self.label_input_descs = list(filter(lambda d: d.is_label, input_descs))

        self.model_input_config = model_input_config
        self.groups = groups
        self.auto_create_embedding = auto_create_embedding
        self.use_fc = len(model_input_config.all_fcs) > 0
        if self.use_fc:
            self._state_mgr = StateManagerImplV2(self, False)
            self.auto_create_embedding = False
            print("{}: input contains feature columns, set auto_create_embedding=False".format(self.name))

        if self.auto_create_embedding:
            self.embedding_layers = self.create_embedding_layers_by_inputs(
                self.feat_input_descs + self.label_input_descs, self.name)
            print("{}: auto created {} embeddding layers: {}"
                  .format(self.name, len(self.embedding_layers), self.embedding_layers))

            self.multihot_layers = self.create_multihot_layers_by_inputs(
                self.feat_input_descs + self.label_input_descs, self.name)
            print("{}: auto created {} multihot layers: {}".
                  format(self.name, len(self.multihot_layers), self.multihot_layers))
        else:
            self.embedding_layers = {}
            self.multihot_layers = {}

    @classmethod
    def create_embedding_layers_by_inputs(cls, input_descs: List[InputDesc], base_name=None):
        embedding_layers = {}
        for ui_desc in input_descs:
            if ui_desc.need_embedding():
                if ui_desc.embedding_name in embedding_layers:
                    continue
                embedding_layers[ui_desc.embedding_name] = VocabEmbeddingLayer(
                    ui_desc.embedding_dim,
                    ui_desc.vocab_size,
                    ui_desc.get_vocab(),
                    None,
                    ui_desc.dtype,
                    ui_desc.hash_type,
                    ui_desc.num_oov_buckets,
                    embedding_initializer=ui_desc.embedding_initializer,
                    embedding_l2_reg=ui_desc.embedding_l2_reg,
                    embedding_l1_reg=ui_desc.embedding_l1_reg,
                    embedding_combiner=ui_desc.embedding_combiner or "mean",
                    name=((base_name + '/') if base_name else '') + ui_desc.embedding_name + "_embedding_layer",
                    max_len=ui_desc.max_len,
                    padding=ui_desc.padding
                )
        return embedding_layers

    @classmethod
    def create_multihot_layers_by_inputs(cls, input_descs: List[InputDesc], base_name=None):
        multihot_layers = {}
        for ui_desc in input_descs:
            if ui_desc.is_one_hot():
                multihot_layers[ui_desc.name] = VocabMultiHotLayer(
                    ui_desc.vocab_size,
                    ui_desc.get_vocab(),
                    None,
                    ui_desc.dtype,
                    ui_desc.hash_type,
                    ui_desc.num_oov_buckets,
                    name=((base_name + '/') if base_name else '') + ui_desc.name + "_multihot_layer"
                )
        return multihot_layers

    def build(self, input_shape):
        if self.use_fc:
            for fc in self.model_input_config.all_fcs:
                fc.create_state(self._state_mgr)

        for emb_layer in self.embedding_layers.values():
            emb_layer.build(input_shape)

        for mhot_layer in self.multihot_layers.values():
            mhot_layer.build(input_shape)

        super(ModelInputLayer, self).build(input_shape)

    def get_embedding_layers(self):
        return self.embedding_layers

    def get_input_config(self):
        return self.model_input_config

    # def get_embedding_layer_by_name(self, embedding_name) -> VocabEmbeddingLayer:
    #     return self.embedding_layers.get(embedding_name)
    def get_embedding_layer_by_name(self, embedding_name):
        return PoolingProcessLayer(self.embedding_layers.get(embedding_name))

    def get_feature_input_descs(self, groups=None):
        feat_input_descs = self.feat_input_descs
        if isinstance(groups, (str, list)) and groups:
            call_feat_names = set()
            for i in self.model_input_config.get_inputs_by_group(groups):
                call_feat_names.add(i.name)
            if not call_feat_names:
                raise RuntimeError("{}: found no input descs by groups {}, input layer groups={}"
                                   .format(self.name, groups, self.groups))
            feat_input_descs = [i for i in self.feat_input_descs if i.name in call_feat_names]
        return feat_input_descs

    @tf.function
    def call(self, inputs, **kwargs):
        if not isinstance(inputs, dict):
            raise RuntimeError("{}: only dict inputs are supported, got '{}': {}"
                               .format(self.name, type(inputs), inputs))

        groups = kwargs.get('groups')
        if groups is not None and not isinstance(groups, (str, list)):
            raise RuntimeError("{}: param 'groups' must be a str/list, got {}: {}"
                               .format(self.name, type(groups), groups))
        feat_input_descs = self.get_feature_input_descs(groups)

        inputs_cpy = {}
        weight_vals = {}

        weight_descs = [i for i in feat_input_descs if i.is_weight_col]
        feat_descs = [i for i in feat_input_descs if not i.is_weight_col]

        for input_desc in weight_descs + feat_descs:
            t, is_ragged, ot = input_desc.transform_tf_tensor(inputs[input_desc.name])
            if self.use_fc:
                if is_ragged:
                    t = tf.squeeze(t, axis=1)
                    inputs_cpy[input_desc.name] = t.to_sparse(name=self.name + '_' + input_desc.name + "_to_sparse")
                else:
                    inputs_cpy[input_desc.name] = t
            else:
                if input_desc.is_weight_col:
                    weight_vals[input_desc.name] = t
                    continue
                if self.auto_create_embedding and input_desc.need_embedding():
                    emb_layer = self.embedding_layers[input_desc.embedding_name]

                    if input_desc.weight_col:
                        weights = weight_vals[input_desc.weight_col]
                    elif input_desc.self_weighted:
                        weights = ot
                    else:
                        weights = None
                    # 先使用VocabEmbeddingLayer得到embedding, 然后通过pooling_process得到最终的embedding
                    embedding, weights_proc = emb_layer(t, weights=weights)
                    # 需要注意的是, 不同的input_desc可能使用相同的emb_layer, 但是后处理时使用input_desc自身的embedding_combiner
                    embedding = pooling_process(t, embedding, weights_proc, input_desc.embedding_combiner)

                    if input_desc.embedding_combiner == 'seq_pad' or input_desc.embedding_combiner == 'seq_nopad':
                        # [batch, 1, max_len, dim] -> [batch, max_len, dim]
                        embedding = tf.squeeze(embedding, axis=1, name=input_desc.name + '_embeddings_squeeze')
                    else:  # [batch, 1, dim] -> [batch, dim]
                        embedding = tf.squeeze(embedding, axis=-2, name=input_desc.name + '_embeddings_squeeze')
                    inputs_cpy[input_desc.name] = embedding
                elif self.auto_create_embedding and input_desc.is_one_hot():
                    mhot_layer = self.multihot_layers[input_desc.name]
                    if input_desc.weight_col:
                        weights = weight_vals[input_desc.weight_col]
                    else:
                        weights = None
                    # [batch, 1, dim]
                    mhot = mhot_layer(t, weights=weights)
                    # [batch, dim]
                    mhot = tf.squeeze(mhot, axis=-2, name=input_desc.name + '_mhot_squeeze')
                    inputs_cpy[input_desc.name] = mhot
                else:
                    inputs_cpy[input_desc.name] = t
                    if input_desc.weight_col and input_desc.get_vocab_size() > 0:
                        inputs_cpy[input_desc.weight_col] = weight_vals[input_desc.weight_col]

        if not self.use_fc:
            return inputs_cpy

        fc_cache = fc_lib.FeatureTransformationCache(inputs_cpy)
        input_feat_vals = {}
        for fc in self.model_input_config.all_fcs:
            if isinstance(fc, fc_lib.CategoricalColumn):
                feat_val = fc.get_sparse_tensors(fc_cache, self._state_mgr)
                feat_val = feat_val.id_tensor
            else:
                feat_val = fc.get_dense_tensor(fc_cache, self._state_mgr)
            input_feat_vals[fc.name] = feat_val
        return input_feat_vals

    def compute_mask_histlen(self, inputs, name=None, return_seqlist=False):
        if name is None:
            raise RuntimeError("In compute mask, param 'name' cannot be None.")

        if name not in self.embedding_layers:
            for desc in self.get_feature_input_descs():
                if name == desc.name:
                    embedding_name = desc.embedding_name
                    inputs_desc = desc
                    break
        else:
            embedding_name = name
            for desc in self.get_feature_input_descs():
                if name == desc.embedding_name:
                    inputs_desc = desc
                    break
        if isinstance(inputs, dict):
            inputs = inputs[name]
        t, is_ragged, _ = inputs_desc.transform_tf_tensor(inputs)
        return self.embedding_layers.get(embedding_name).compute_mask_histlen(t, return_seqlist)

    def get_tensor_specs(self, groups=None):
        feat_input_descs = self.get_feature_input_descs(groups)
        call_fn_specs = {d.name: d.to_tf_tensor_spec() for d in feat_input_descs}
        return call_fn_specs

    def get_config(self):
        config = super(ModelInputLayer, self).get_config()
        config.update({
            'model_input_config': self.model_input_config,
            'groups': self.groups,
            'auto_create_embedding': self.auto_create_embedding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
