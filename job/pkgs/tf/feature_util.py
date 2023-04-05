# coding=utf-8
# @Time     : 2021/1/6 17:29
# @Auther   : lionpeng@tencent.com

import json
import os
import traceback
from collections import defaultdict, namedtuple
from itertools import chain
from typing import Dict, List

import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.ops import variable_scope, variables
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures

from ..utils import find_duplicated_entries, recur_expand_param
from .helperfuncs import TF_REF_VERSION

if tf.__version__ < TF_REF_VERSION:
    try:
        from tensorflow.python.keras.mixed_precision.experimental import \
            policy as precision_policy
    except:
        from tensorflow.python.keras.mixed_precision import \
            policy as precision_policy
else:
    from tensorflow.python.keras.mixed_precision import policy as precision_policy

import collections


def concat_func(embedding_dict, flatten=True, dtype_policy=None):
    item_vals = []
    for name, val in embedding_dict.items():
        if flatten:
            val = tf.keras.layers.Flatten()(val)
        if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
            raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                               .format(name, val.dtype))
        if dtype_policy and val.dtype != dtype_policy.compute_dtype:
            val = tf.cast(val, dtype_policy.compute_dtype,
                          name=name + '_cast2' + dtype_policy.compute_dtype)
        item_vals.append(val)
    return item_vals


class StateManagerImpl(fc_lib.StateManager):
    """Manages the state of DenseFeatures and LinearLayer."""

    def __init__(self, layer, trainable):
        """Creates an _StateManagerImpl object.

    Args:
      layer: The input layer this state manager is associated with.
      trainable: Whether by default, variables created are trainable or not.
    """
        self._trainable = trainable
        self._layer = layer
        if self._layer is not None and not hasattr(self._layer, '_resources'):
            self._layer._resources = data_structures.Mapping()  # pylint: disable=protected-access
        self._cols_to_vars_map = collections.defaultdict(lambda: {})
        self._cols_to_resources_map = collections.defaultdict(lambda: {})

    def create_variable(self,
                        feature_column,
                        name,
                        shape,
                        dtype=None,
                        trainable=True,
                        use_resource=True,
                        initializer=None):
        if name in self._cols_to_vars_map[feature_column]:
            raise ValueError('Variable already exists.')

        # We explicitly track these variables since `name` is not guaranteed to be
        # unique and disable manual tracking that the add_weight call does.
        with trackable.no_manual_dependency_tracking_scope(self._layer):
            var = self._layer.add_weight(
                name=name,
                shape=shape,
                dtype=dtype,
                initializer=initializer,
                trainable=self._trainable and trainable,
                use_resource=use_resource,
                # TODO(rohanj): Get rid of this hack once we have a mechanism for
                # specifying a default partitioner for an entire layer. In that case,
                # the default getter for Layers should work.
                getter=variable_scope.get_variable)
        if isinstance(var, variables.PartitionedVariable):
            for v in var:
                part_name = name + '/' + str(v._get_save_slice_info().var_offset[0])  # pylint: disable=protected-access
                self._layer._track_trackable(v,
                                             feature_column.name + '/' + part_name)  # pylint: disable=protected-access
        else:
            if isinstance(var, trackable.Trackable):
                self._layer._track_trackable(var, feature_column.name + '/' + name)  # pylint: disable=protected-access

        self._cols_to_vars_map[feature_column][name] = var
        return var

    def get_variable(self, feature_column, name):
        if name in self._cols_to_vars_map[feature_column]:
            return self._cols_to_vars_map[feature_column][name]
        raise ValueError('Variable does not exist.')

    def add_resource(self, feature_column, resource_name, resource):
        self._cols_to_resources_map[feature_column][resource_name] = resource
        # pylint: disable=protected-access
        if self._layer is not None and isinstance(resource, trackable.Trackable):
            # Add trackable resources to the layer for serialization.
            if feature_column.name not in self._layer._resources:
                self._layer._resources[feature_column.name] = data_structures.Mapping()
            if resource_name not in self._layer._resources[feature_column.name]:
                self._layer._resources[feature_column.name][resource_name] = resource
        # pylint: enable=protected-access

    def has_resource(self, feature_column, resource_name):
        return resource_name in self._cols_to_resources_map[feature_column]

    def get_resource(self, feature_column, resource_name):
        if (feature_column not in self._cols_to_resources_map or
                resource_name not in self._cols_to_resources_map[feature_column]):
            raise ValueError('Resource does not exist.')
        return self._cols_to_resources_map[feature_column][resource_name]


class StateManagerImplV2(StateManagerImpl):
    """Manages the state of DenseFeatures."""

    def create_variable(self,
                        feature_column,
                        name,
                        shape,
                        dtype=None,
                        trainable=True,
                        use_resource=True,
                        initializer=None):
        if name in self._cols_to_vars_map[feature_column]:
            raise ValueError('Variable already exists.')

        # We explicitly track these variables since `name` is not guaranteed to be
        # unique and disable manual tracking that the add_weight call does.
        with trackable.no_manual_dependency_tracking_scope(self._layer):
            var = self._layer.add_weight(
                name=name,
                shape=shape,
                dtype=dtype,
                initializer=initializer,
                trainable=self._trainable and trainable,
                use_resource=use_resource)
        if isinstance(var, trackable.Trackable):
            self._layer._track_trackable(var, feature_column.name + '/' + name)  # pylint: disable=protected-access
        self._cols_to_vars_map[feature_column][name] = var
        return var


def dtype_from_str(dtype_str) -> tf.dtypes.DType:
    if not dtype_str:
        return None
    dtype_str = str(dtype_str).strip().lower()
    if dtype_str in ['int', 'int32']:
        return tf.int32
    if dtype_str in ['int64']:
        return tf.int64
    if dtype_str in ['int16']:
        return tf.int16
    if dtype_str in ['int8']:
        return tf.int8
    if dtype_str in ['float', 'float32']:
        return tf.float32
    if dtype_str in ['float64']:
        return tf.float64
    if dtype_str in ['float16']:
        return tf.float16
    if dtype_str in ['str', 'string']:
        return tf.string
    return None


class InputDesc(namedtuple('InputDesc',
                           ['name', 'dtype_str', 'shape', 'is_sparse', 'is_ragged', 'default_value', 'val_sep',
                            'is_label', 'embedding_dim', 'vocab_size', 'vocab_list', 'vocab_file', 'embedding_name',
                            'embedding_combiner', 'embedding_l1_reg', 'embedding_l2_reg', 'embedding_initializer',
                            'is_varlen', 'one_hot', 'bucket_boundaries', 'is_sample_weight', 'label_index',
                            'hash_type', 'zscore', 'std', 'mean', 'weight_col', 'exclude', 'max_len', 'padding',
                            'num_oov_buckets', 'self_weighted', 'dynamic_embedding'])):

    def __new__(cls, name, dtype_str, shape=(1,), is_sparse=False, is_ragged=False, default_value=None,
                val_sep=None, is_label=False, embedding_dim=None, vocab_size=None, vocab_list=None,
                vocab_file=None, embedding_name=None, embedding_combiner='mean', embedding_l1_reg=None,
                embedding_l2_reg=None, embedding_initializer='uniform', is_varlen=False, one_hot=False,
                bucket_boundaries=None, is_sample_weight=False, label_index=None, hash_type=None,
                zscore=False, std=None, mean=None, weight_col=None, exclude=False, max_len=None, padding=None,
                num_oov_buckets=None, self_weighted=False, dynamic_embedding=False):
        return super(InputDesc, cls).__new__(cls, name, dtype_str, shape, is_sparse, is_ragged, default_value,
                                             val_sep, is_label, embedding_dim, vocab_size, vocab_list,
                                             vocab_file, embedding_name or name + "_embedding", embedding_combiner,
                                             embedding_l1_reg, embedding_l2_reg, embedding_initializer, is_varlen,
                                             one_hot, bucket_boundaries, is_sample_weight, label_index, hash_type,
                                             zscore, std, mean, weight_col, exclude, max_len, padding, num_oov_buckets,
                                             self_weighted, dynamic_embedding)

    def __init__(self, *args, **kwargs):
        self._vocab_from_file = None
        if self.vocab_file:
            vocab_list = []
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    if self.dtype.is_integer:
                        vocab_list.append(int(line))
                    else:
                        vocab_list.append(line)
            if not vocab_list:
                raise RuntimeError("vocab file '{}' of input '{}' is empty".format(self.vocab_file, self.name))
            if len(vocab_list) != len(set(vocab_list)):
                duplicated_entries = find_duplicated_entries(vocab_list)
                raise RuntimeError("duplicated entries in 'vocab_file' '{}' of input '{}': {}"
                                   .format(self.vocab_file, self.name, duplicated_entries))
            self._vocab_from_file = sorted(vocab_list)
            print("read {} words from vocab file '{}' of input '{}'".format(len(self._vocab_from_file),
                                                                            self.vocab_file, self.name))
        self.is_weight_col = False

    @property
    def dtype(self):
        return dtype_from_str(self.dtype_str)

    def get_vocab(self):
        if self.dynamic_embedding:
            return None
        if self.vocab_list:
            return self.vocab_list
        if self._vocab_from_file:
            return self._vocab_from_file
        return None

    def get_vocab_size(self):
        if self.dynamic_embedding:
            return -1
        if self.vocab_size is not None and self.vocab_size > 0:
            return self.vocab_size
        vocab = self.get_vocab()
        if vocab is None:
            return 0
        if self.num_oov_buckets:
            return len(vocab)+self.num_oov_buckets
        return len(vocab)

    def need_embedding(self):
        return self.embedding_dim is not None and self.embedding_dim > 0 and \
            (self.get_vocab_size() > 0 or self.dynamic_embedding)

    def is_one_hot(self):
        return self.one_hot and self.get_vocab_size() > 0

    @classmethod
    def parse_from_json(cls, conf_json):
        name = conf_json.get('name')
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("'name' of input must be a non-empty string, got '{}': {}"
                               .format(type(name), name))
        name = name.strip()
        dtype = dtype_from_str(conf_json.get('dtype'))
        if dtype is None:
            raise RuntimeError("unknown dtype '{}' of input '{}', conf_json={}"
                               .format(conf_json.get('dtype'), name, conf_json))

        default_value = conf_json.get('default_value')
        val_sep = conf_json.get('val_sep')
        is_label = conf_json.get('is_label', False)
        embedding_dim = conf_json.get('embedding_dim')
        vocab_size = conf_json.get('vocab_size')
        if isinstance(vocab_size, int) and vocab_size <= 0:
            raise RuntimeError("invalid 'vocab_size' {} of input '{}', should be > 0".format(vocab_size, name))
        elif not isinstance(vocab_size, int):
            vocab_size = None

        vocab_list = conf_json.get('vocab_list')
        if isinstance(vocab_list, (tuple, list)) and vocab_list:
            if vocab_size:
                raise RuntimeError("intput '{}': only one of 'vocab_size', 'vocab_list', 'vocab_file'"
                                   " can be specified".format(name))
            if len(vocab_list) != len(set(vocab_list)):
                duplicated_entries = find_duplicated_entries(vocab_list)
                raise RuntimeError("duplicated entries in 'vocab_list' of input '{}': {}"
                                   .format(name, duplicated_entries))
            vocab_list = sorted(vocab_list)
        else:
            vocab_list = None

        vocab_file = conf_json.get('vocab_file')
        if isinstance(vocab_file, str) and vocab_file.strip():
            if vocab_size or vocab_list:
                raise RuntimeError("intput '{}': only one of 'vocab_size', 'vocab_list', 'vocab_file'"
                                   " can be specified".format(name))
            vocab_file = vocab_file.strip()
            if not os.path.isfile(vocab_file):
                raise RuntimeError("'vocab_file' '{}' of input '{}' not exist".format(vocab_file, name))
        else:
            vocab_file = None

        embedding_name = conf_json.get('embedding_name', '').strip()
        embedding_combiner = conf_json.get('embedding_combiner', 'mean').strip().lower()
        embedding_l1_reg = conf_json.get('embedding_l1_reg')
        embedding_l2_reg = conf_json.get('embedding_l2_reg')
        embedding_initializer = conf_json.get('embedding_initializer', 'uniform')
        is_varlen = conf_json.get('is_varlen', False)
        is_sparse = conf_json.get('is_sparse', False)
        is_ragged = conf_json.get('is_ragged', False)
        if is_sparse and is_ragged:
            raise RuntimeError("only one of 'is_sparse' and 'is_ragged' can be True, but got both True of input '{}',"
                               " conf_json={}".format(name, conf_json))
        shape = conf_json.get('shape')
        if not shape:
            if is_sparse or is_ragged:
                shape = (None,)
            else:
                shape = (1,)
            print("'shape' of input '{}' not specified, assume it's to be {}, conf_json={}"
                  .format(name, shape, conf_json))
        elif not isinstance(shape, (tuple, list)):
            shape = (int(shape),)
        else:
            shape = tuple(shape)
        one_hot = conf_json.get('one_hot', False)
        if one_hot and embedding_dim is not None and embedding_dim > 0:
            raise RuntimeError("input '{}' can only be one-hot or embedded, not both".format(name))

        bucket_boundaries = conf_json.get('bucket_boundaries')
        if bucket_boundaries:
            if not isinstance(bucket_boundaries, (tuple, list)) or \
                    not all([isinstance(b, (int, float)) for b in bucket_boundaries]) or \
                    len(bucket_boundaries) != len(set(bucket_boundaries)):
                raise RuntimeError("'bucket_boundaries' of input '{}' should be a list with unique numbers, got: {}"
                                   .format(name, bucket_boundaries))
            bucket_boundaries = sorted(bucket_boundaries)
            vocab_size = len(bucket_boundaries) + 1
            vocab_list = None
            vocab_file = None

        is_sample_weight = conf_json.get('is_sample_weight', False)
        label_index = conf_json.get('label_index')
        hash_type = conf_json.get('hash_type', '').strip().lower()
        if hash_type and hash_type not in ['hash', 'mod']:
            raise RuntimeError("'hash_type' of input '{}' can only be 'hash'/'mod', got '{}'"
                               .format(name, hash_type))

        zscore = conf_json.get('zscore', False)
        std = conf_json.get('std')
        mean = conf_json.get('mean')

        if zscore and (std is None or std < 0 or mean is None):
            raise RuntimeError("input '{}': must provide valid 'std' and 'mean' value when using zscore,"
                               " got std={}, mean={}".format(name, std, mean))

        weight_col = conf_json.get("weight_col", '').strip()
        if weight_col == name:
            raise RuntimeError("input '{}': 'weight_col' can not be self".format(name))

        exclude = conf_json.get('exclude', False)
        max_len = conf_json.get('max_len', False)
        padding = conf_json.get('padding', False)
        num_oov_buckets = conf_json.get('num_oov_buckets', None)
        self_weighted = conf_json.get('self_weighted', False)
        if self_weighted and not dtype.is_floating and not dtype.is_integer:
            raise RuntimeError("self weighted column '{}' must be float/integer column, got {}".format(name, dtype))
        dynamic_embedding = conf_json.get('dynamic_embedding', False)

        return InputDesc(name, conf_json.get('dtype'), shape, is_sparse, is_ragged, default_value, val_sep, is_label,
                         embedding_dim, vocab_size, vocab_list, vocab_file, embedding_name, embedding_combiner,
                         embedding_l1_reg, embedding_l2_reg, embedding_initializer, is_varlen, one_hot,
                         bucket_boundaries, is_sample_weight, label_index, hash_type, zscore, std, mean, weight_col,
                         exclude, max_len, padding, num_oov_buckets, self_weighted, dynamic_embedding)

    def infer_default_value(self):
        if self.default_value is not None:
            return tf.constant(self.default_value, dtype=self.dtype)
        if self.val_sep or self.dtype == tf.string:
            return tf.constant('', dtype=tf.string)
        return tf.constant(0, dtype=self.dtype)

    def to_tf_input(self):
        if not self.val_sep:
            tf_input = tf.keras.Input(shape=self.shape, name=self.name, dtype=self.dtype,
                                      sparse=self.is_sparse, ragged=self.is_ragged)
        else:
            tf_input = tf.keras.Input(shape=(1,), name=self.name, dtype=tf.string)
        return tf_input

    def to_tf1_placeholder(self):
        shape = [None] + list(self.shape)
        if not self.val_sep:
            tf_ph = tf.compat.v1.placeholder(dtype=self.dtype, shape=shape, name=self.name)
        else:
            tf_ph = tf.compat.v1.placeholder(dtype=tf.string, shape=[None, 1], name=self.name)
        return tf_ph

    def to_tf_tensor_spec(self):
        if self.is_sparse:
            return tf.SparseTensorSpec(shape=None, dtype=self.dtype)
        if self.is_ragged:
            return tf.RaggedTensorSpec(shape=None, dtype=self.dtype)

        return tf.TensorSpec(shape=(None,)+self.shape,
                             dtype=tf.string if self.val_sep else self.dtype,
                             name=self.name)

    def transform_tf_tensor(self, tensor: tf.Tensor):
        is_ragged = False
        if self.val_sep:
            if tensor.dtype != tf.string:
                tensor = tf.as_string(tensor, name=self.name + "_as_string")
            # ragged: [batch, 1, None]
            tensor = tf.strings.split(tensor, sep=self.val_sep, name=self.name + "_splitted")
            if self.dtype != tf.string:
                tensor = tf.strings.to_number(tensor, out_type=self.dtype, name=self.name + "_to_number")
            is_ragged = True

        if self.zscore:
            if not tensor.dtype.is_floating and not tensor.dtype.is_integer:
                raise RuntimeError("dtype of '{}' is {}, only numberical feature support zscore transform"
                                   .format(self.name, self.dtype))
            if not tensor.dtype.is_floating:
                tensor = tf.cast(tensor, tf.float32, name=self.name+"_int2float")
            # tensor = (tensor - self.mean)/(self.std + 1e-8)
            tensor = tf.divide(tf.subtract(tensor, self.mean), tf.add(self.std, 1e-8))
        orig_tensor = tensor

        if self.bucket_boundaries:
            if not tensor.dtype.is_floating and not tensor.dtype.is_integer:
                raise RuntimeError("dtype of '{}' is {}, only numberical feature support bucketize"
                                   .format(self.name, self.dtype))
            if is_ragged:
                tensor = tf.ragged.map_flat_values(tf.raw_ops.Bucketize, input=tensor,
                                                   boundaries=self.bucket_boundaries)
            else:
                tensor = tf.raw_ops.Bucketize(input=tensor, boundaries=self.bucket_boundaries,
                                              name=self.name + '_bucketize')

        return tensor, is_ragged, orig_tensor

    def get_config(self):
        config = dict(self._asdict())
        if self.vocab_file:
            # 去掉对vocab file的依赖
            config['vocab_file'] = None
            config['vocab_list'] = self._vocab_from_file
        config['is_weight_col'] = self.is_weight_col
        return config

    @classmethod
    def from_config(cls, config):
        is_weight_col = config.pop('is_weight_col', False)
        ins = cls(**config)
        ins.is_weight_col = is_weight_col
        return ins


class FeatureProcessDesc(namedtuple('FeatureProcessDesc', ['op', 'inputs', 'outputs', 'args'])):
    OP_MAP = {
        "bucket": (tf.feature_column.bucketized_column, ('boundaries',)),
        "hash_bucket": (tf.feature_column.categorical_column_with_hash_bucket, ('hash_bucket_size',)),
        "identity": (tf.feature_column.categorical_column_with_identity, ('num_buckets',)),
        "vocab_list": (tf.feature_column.categorical_column_with_vocabulary_list, ('vocabulary_list',)),
        "vocab_file": (tf.feature_column.categorical_column_with_vocabulary_file, ('vocabulary_file',)),
        "cross": (tf.feature_column.crossed_column, ('hash_bucket_size',)),
        "embedding": (tf.feature_column.embedding_column, ('dimension',)),
        "share_embedding": (tf.feature_column.shared_embeddings, ('dimension',)),
        "numeric": (tf.feature_column.numeric_column, ()),
        "indicator": (tf.feature_column.indicator_column, ())
    }

    def __new__(cls, op, inputs=None, outputs=None, args={}):
        return super(FeatureProcessDesc, cls).__new__(cls, op, inputs, outputs, args)

    def to_tf_fcs(self, model_input_descs_dict: Dict[str, InputDesc], parsed_fcs_dict: Dict):
        if self.op == tf.feature_column.bucketized_column:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            boundaries = self.args.get('boundaries')
            if not isinstance(boundaries, (tuple, list)) or not boundaries:
                raise RuntimeError("'boundaries' should be a non-empty tuple/list for fc op '{}', got '{}': {}"
                                   .format(self.op, type(boundaries), boundaries))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                if model_input_desc.dtype.is_floating or model_input_desc.dtype.is_integer:
                    num_fc = tf.feature_column.numeric_column(self.inputs, shape=model_input_desc.shape,
                                                              default_value=model_input_desc.default_value,
                                                              dtype=model_input_desc.dtype)
                    print("auto wrapped input '{}' with numeric_column before apply bucketize".format(self.inputs))
                    return self.op(num_fc, boundaries)
                else:
                    raise RuntimeError("dtype {} of input '{}' is not numerical, cannot do bucketize, input desc={}"
                                       .format(model_input_desc.dtype, self.inputs, model_input_desc))
            else:
                num_fc = parsed_fcs_dict.get(self.inputs)
                if num_fc is None:
                    raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
                return self.op(num_fc, boundaries)
        elif self.op == tf.feature_column.categorical_column_with_hash_bucket:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                hash_bucket_size = self.args.get('hash_bucket_size')
                if not isinstance(hash_bucket_size, int) or hash_bucket_size <= 0:
                    if model_input_desc.get_vocab_size() <= 0:
                        raise RuntimeError("'hash_bucket_size' should be a positive int for fc op '{}', got '{}': {}"
                                           .format(self.op, type(hash_bucket_size), hash_bucket_size))
                    hash_bucket_size = model_input_desc.get_vocab_size()
                    print("'hash_bucket_size' of fc op '{}' not set, use vocab size {} as buckets size"
                          .format(self.op, hash_bucket_size))
                args = {'dtype': model_input_desc.dtype}
                arg_dtype = dtype_from_str(self.args.get('dtype', ''))
                if arg_dtype is not None:
                    args = {'dtype': arg_dtype}
                return self.op(model_input_desc.name, hash_bucket_size, **args)
            else:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
        elif self.op == tf.feature_column.categorical_column_with_identity:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                num_buckets = self.args.get('num_buckets')
                if not isinstance(num_buckets, int) or num_buckets <= 0:
                    if model_input_desc.get_vocab_size() <= 0:
                        raise RuntimeError("'num_buckets' should be a positive int for fc op '{}', got '{}': {}"
                                           .format(self.op, type(num_buckets), num_buckets))
                    num_buckets = model_input_desc.get_vocab_size()
                    print("'num_buckets' of fc op '{}' not set, use vocab size {} as buckets size"
                          .format(self.op, num_buckets))
                args = {'default_value': model_input_desc.default_value}
                arg_default_value = self.args.get('default_value')
                if arg_default_value is not None:
                    args = {'default_value': arg_default_value}
                return self.op(model_input_desc.name, num_buckets, **args)
            else:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
        elif self.op == tf.feature_column.categorical_column_with_vocabulary_list:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            vocabulary_list = self.args.get('vocabulary_list')
            if not isinstance(vocabulary_list, (tuple, list)) or not vocabulary_list:
                raise RuntimeError("'vocabulary_list' should be a non-empty list/tuple for fc op '{}', got '{}': {}"
                                   .format(self.op, type(vocabulary_list), vocabulary_list))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                args = {'dtype': model_input_desc.dtype}
                if model_input_desc.default_value is not None:
                    args['default_value'] = model_input_desc.default_value
                args.update(self.args)
                return self.op(model_input_desc.name, **args)
            else:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
        elif self.op == tf.feature_column.categorical_column_with_vocabulary_file:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            vocabulary_file = self.args.get('vocabulary_file')
            if not isinstance(vocabulary_file, str):
                raise RuntimeError("'vocabulary_file' should be a path to existed file for fc op '{}', got '{}': {}"
                                   .format(self.op, type(vocabulary_file), vocabulary_file))
            if not os.path.isfile(vocabulary_file):
                raise RuntimeError("'vocabulary_file' '{}' not exist for fc op '{}'".format(vocabulary_file, self.op))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                args = {'dtype': model_input_desc.dtype, 'default_value': model_input_desc.default_value}
                args.update(self.args)
                args.pop('vocabulary_file', None)
                return self.op(model_input_desc.name, vocabulary_file, **args)
            else:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
        elif self.op == tf.feature_column.crossed_column:
            if not isinstance(self.inputs, (list, tuple)) or len(self.inputs) <= 1:
                raise RuntimeError("'in' of fc op '{}' should be a tuple/list with at least 2 elements, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            hash_bucket_size = self.args.get('hash_bucket_size')
            if not isinstance(hash_bucket_size, int) or hash_bucket_size <= 0:
                raise RuntimeError("'hash_bucket_size' should be a positive int for fc op '{}', got '{}': {}"
                                   .format(self.op, type(hash_bucket_size), hash_bucket_size))
            crossed_cols = []
            for i in self.inputs:
                if i in model_input_descs_dict:
                    crossed_cols.append(i)
                else:
                    fc = parsed_fcs_dict.get(i)
                    if fc is None:
                        raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(i, self.op))
                    crossed_cols.append(fc)
            return self.op(crossed_cols, hash_bucket_size)
        elif self.op == tf.feature_column.embedding_column:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            dimension = self.args.get('dimension')
            if not isinstance(dimension, int) or dimension <= 0:
                raise RuntimeError("'dimension' should be a positive int for fc op '{}', got '{}': {}"
                                   .format(self.op, type(dimension), dimension))
            fc = parsed_fcs_dict.get(self.inputs)
            if fc is None:
                input_desc = model_input_descs_dict.get(self.inputs)
                if input_desc is None or input_desc.get_vocab_size() <= 0:
                    raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
                print("found no fc def of input '{}' of fc op '{}', try auto wrap it with"
                      " categorical_column_with_hash_bucket".format(self.inputs, self.op))
                fc = tf.feature_column.categorical_column_with_hash_bucket(self.inputs, input_desc.get_vocab_size(),
                                                                           dtype=input_desc.dtype)
            args = self.args.copy()
            args.pop('dimension', None)
            args.pop('initializer', None)
            args.pop('ckpt_to_load_from', None)
            args.pop('tensor_name_in_ckpt', None)
            return self.op(fc, dimension, **args)
        elif self.op == tf.feature_column.shared_embeddings:
            if not isinstance(self.inputs, (list, tuple)) or not self.inputs:
                raise RuntimeError("'in' of fc op '{}' should be a non-empty tuple/list, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, (list, tuple)) or len(self.outputs) != len(self.inputs):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty tuple/list with same size with 'in',"
                                   " got '{}': {}, in={}".format(self.op, type(self.outputs), self.outputs,
                                                                 self.inputs))
            dimension = self.args.get('dimension')
            if not isinstance(dimension, int) or dimension <= 0:
                raise RuntimeError("'dimension' should be a positive int for fc op '{}', got '{}': {}"
                                   .format(self.op, type(dimension), dimension))
            fcs = []
            for i in self.inputs:
                fc = parsed_fcs_dict.get(i)
                if fc is None:
                    input_desc = model_input_descs_dict.get(i)
                    if input_desc is None or input_desc.get_vocab_size() <= 0:
                        raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(i, self.op))
                    print("found no fc def of input '{}' of fc op '{}', try auto wrap it with"
                          " categorical_column_with_hash_bucket".format(i, self.op))
                    fc = tf.feature_column.categorical_column_with_hash_bucket(i, input_desc.get_vocab_size(),
                                                                               dtype=input_desc.dtype)
                fcs.append(fc)
            args = self.args.copy()
            args.pop('dimension', None)
            args.pop('initializer', None)
            args.pop('ckpt_to_load_from', None)
            args.pop('tensor_name_in_ckpt', None)
            return self.op(fcs, dimension, **args)
        elif self.op == tf.feature_column.numeric_column:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            if self.inputs in model_input_descs_dict:
                model_input_desc = model_input_descs_dict.get(self.inputs)
                args = {'shape': model_input_desc.shape,
                        'default_value': model_input_desc.default_value}
                args.update(self.args)
                arg_dtype = dtype_from_str(self.args.get('dtype', ''))
                if arg_dtype is not None:
                    args['dtype'] = arg_dtype
                else:
                    args['dtype'] = model_input_desc.dtype
                return self.op(self.inputs, **args)
            else:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
        elif self.op == tf.feature_column.indicator_column:
            if not isinstance(self.inputs, str):
                raise RuntimeError("'in' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.inputs), self.inputs))
            if not isinstance(self.outputs, str):
                raise RuntimeError("'out' of fc op '{}' should be a non-empty str, got '{}': {}"
                                   .format(self.op, type(self.outputs), self.outputs))
            fc = parsed_fcs_dict.get(self.inputs)
            if fc is None:
                raise RuntimeError("fc input '{}' of fc op '{}' is not found".format(self.inputs, self.op))
            return self.op(fc)
        else:
            raise NotImplementedError("unsupported fc op '{}'".format(self.op))

    @classmethod
    def parse_from_json(cls, conf_json):
        op = conf_json.get('op')
        if not op or op not in cls.OP_MAP:
            raise RuntimeError("unsupported fc op '{}'".format(op))
        op, mandatory_args = cls.OP_MAP.get(op)
        inputs = conf_json.get('in')
        if inputs is None:
            raise RuntimeError("'in' of fc op '{}' not set!".format(op))
        if isinstance(inputs, str):
            inputs = inputs.strip()
            if not inputs:
                raise RuntimeError("'in' of fc op '{}' is empty!".format(op))
        elif isinstance(inputs, (list, tuple)):
            if not inputs:
                raise RuntimeError("'in' of fc op '{}' is empty!".format(op))
            if not all(isinstance(i, str) for i in inputs):
                raise RuntimeError("'in' of fc op '{}' contains non-string elements, got: {}".format(op, inputs))
            inputs = [i.strip() for i in inputs]
            if not all(inputs):
                raise RuntimeError("'in' of fc op '{}' contains empty strings, got: {}".format(op, inputs))
        else:
            raise RuntimeError("'in' of fc op '{}' should be a non-empty str/tuple/list, got '{}': {}"
                               .format(op, type(inputs), inputs))

        outputs = conf_json.get('out', '')
        if isinstance(outputs, str):
            outputs = outputs.strip()
        elif isinstance(outputs, (tuple, list)):
            if not outputs:
                raise RuntimeError("'out' of fc op '{}' can not be an empty list/tuple".format(op))
            if not all(isinstance(o, str) for o in outputs):
                raise RuntimeError("'out' of fc op '{}' contains non-string elements, got: {}".format(op, outputs))
            outputs = [o.strip() for o in outputs]
            if not all(outputs):
                raise RuntimeError("'out' of fc op '{}' contains empty strings, got: {}".format(op, outputs))
            if not isinstance(inputs, (list, tuple)):
                raise RuntimeError("'out' of fc op '{}' is a tuple/list, but 'in' is not, got in={}, out={}"
                                   .format(op, inputs, outputs))
            if len(outputs) != len(set(outputs)):
                raise RuntimeError("'out' of fc op '{}' contains duplicated names, got {}".format(op, outputs))
            if len(inputs) != len(outputs):
                raise RuntimeError("'in' size {} != 'out' size {} of fc op '{}'".format(len(inputs), len(outputs), op))
        elif outputs is not None:
            raise RuntimeError("'outputs' of fc op '{}' should be a None/str/tuple/list, got '{}': {}"
                               .format(op, type(outputs), outputs))

        args = conf_json.get('args', {})
        if not isinstance(args, dict):
            raise RuntimeError("'args' of fc op '{}' should be None/dict, got '{}': {}".format(op, type(args), args))
        if not set(mandatory_args).issubset(args.keys()):
            raise RuntimeError("'args' of fc op '{}' should at least contain args {}, got {}"
                               .format(op, mandatory_args, args))
        return FeatureProcessDesc(op, inputs, outputs, args)


class ModelInputConfig(object):
    __slots__ = ['_grouped_inputs', '_grouped_fcs', '_input_index', '_fc_index', '_kv_config_file']
    DEF_FEATURE_GROUP = "def_group"

    def __init__(self, grouped_inputs: Dict[str, List[InputDesc]],
                 grouped_fcs: Dict[str, List[fc_lib.FeatureColumn]] = None,
                 kv_config_file: str = None):
        self._grouped_inputs = grouped_inputs
        self._grouped_fcs = grouped_fcs

        self._input_index = {}
        for gn, ins in grouped_inputs.items():
            for i in ins:
                self._input_index[i.name] = i

        self._fc_index = {}
        if grouped_fcs:
            for gn, fcs in grouped_fcs.items():
                for fc in fcs:
                    self._fc_index[fc.name] = fc
        self._kv_config_file = kv_config_file

    @classmethod
    def _parse_inputs(cls, inputs_json):
        grouped_inputs = {}
        if not isinstance(inputs_json, (list, tuple, dict)) or not inputs_json:
            raise RuntimeError("'inputs' should be a non-empty list/tuple/dict, got '{}': {}"
                               .format(type(inputs_json), inputs_json))

        all_inputs = dict()
        label_index_set = set()
        col_weight_pair = dict()

        def __parse_input_group(group_name, input_list):
            grouped_inputs[group_name] = []
            for i, cfg in enumerate(input_list):
                input_desc = InputDesc.parse_from_json(cfg)
                if input_desc.name in all_inputs:
                    raise RuntimeError(
                        "input name '{}' of group '{}' is duplicated".format(input_desc.name, group_name))
                if input_desc.is_label and input_desc.label_index is not None:
                    if input_desc.label_index in label_index_set:
                        raise RuntimeError("label_index {} of input '{}' of group '{}' duplicated"
                                           .format(input_desc.label_index, input_desc.name, group_name))
                    label_index_set.add(input_desc.label_index)
                all_inputs[input_desc.name] = input_desc
                grouped_inputs[group_name].append(input_desc)
                if input_desc.weight_col:
                    col_weight_pair[input_desc.name] = input_desc.weight_col

        if isinstance(inputs_json, (list, tuple)):
            __parse_input_group(cls.DEF_FEATURE_GROUP, inputs_json)
        else:
            for gname, ilist in inputs_json.items():
                if gname == cls.DEF_FEATURE_GROUP:
                    raise RuntimeError("input group name '{}' is reserved, please change group name".format(gname))
                __parse_input_group(gname, ilist)

        label_count = len([i for i in list(chain.from_iterable(grouped_inputs.values())) if i.is_label])
        if label_index_set and len(label_index_set) != label_count:
            raise RuntimeError("number of label_index {} != number of label {}"
                               .format(len(label_index_set), label_count))

        for col, weight_col in col_weight_pair.items():
            wc_input = all_inputs.get(weight_col)
            if wc_input is None:
                raise RuntimeError("'weight_col' '{}' of input '{}' not exists".format(weight_col, col))
            assert not wc_input.weight_col, "'weight_col' '{}' cannot set other 'weight_col' '{}'"\
                .format(wc_input.name, wc_input.weight_col)
            wc_input.is_weight_col = True

        return grouped_inputs

    @classmethod
    def _parse_inputs_v2(cls, inputs_json):
        if not isinstance(inputs_json, dict) or not inputs_json:
            raise RuntimeError("'inputs' should be a non-empty dict, got '{}': {}"
                               .format(type(inputs_json), inputs_json))

        inputs_defs = inputs_json.get('defs', inputs_json.get('defines'))
        if not isinstance(inputs_defs, (tuple, list)) or not inputs_defs:
            raise RuntimeError("'defs/defines' in 'inputs' should be a non-empty tuple/list, got '{}': {}"
                               .format(type(inputs_defs), inputs_defs))

        all_inputs = dict()
        label_index_set = set()
        col_weight_pair = dict()

        for i, cfg in enumerate(inputs_defs):
            input_desc = InputDesc.parse_from_json(cfg)
            if input_desc.name in all_inputs:
                raise RuntimeError(
                    "input name '{}' duplicated".format(input_desc.name))
            if input_desc.is_label and input_desc.label_index is not None:
                if input_desc.label_index in label_index_set:
                    raise RuntimeError("label_index {} of input '{}' duplicated"
                                       .format(input_desc.label_index, input_desc.name))
                label_index_set.add(input_desc.label_index)
            all_inputs[input_desc.name] = input_desc
            if input_desc.weight_col:
                col_weight_pair[input_desc.name] = input_desc.weight_col

        for col, weight_col in col_weight_pair.items():
            wc_input = all_inputs.get(weight_col)
            if wc_input is None:
                raise RuntimeError("'weight_col' '{}' of input '{}' not exists".format(weight_col, col))
            assert not wc_input.weight_col, "'weight_col' '{}' cannot set other 'weight_col' '{}'" \
                .format(wc_input.name, wc_input.weight_col)
            wc_input.is_weight_col = True

        grouped_inputs = defaultdict(list)
        groups = inputs_json.get("groups")
        if not groups:
            grouped_inputs[cls.DEF_FEATURE_GROUP] = list(all_inputs.values())
        else:
            if not isinstance(groups, dict):
                raise RuntimeError("'groups' of 'inputs' should be a dict, got '{}': {}".format(type(groups), groups))
            for gn, input_names in groups.items():
                if not isinstance(gn, str) or not gn.strip():
                    raise RuntimeError("key in 'groups' of 'inputs' should be non-empty strings, got '{}': {}"
                                       .format(type(gn), gn))
                gn = gn.strip()
                if not isinstance(input_names, (tuple, list)) or not input_names:
                    raise RuntimeError("value in inputs group '{}' should be non-empty tuple/lists, got '{}': {}"
                                       .format(gn, type(input_names), input_names))
                if not all([n.strip() for n in input_names]):
                    raise RuntimeError("names in inputs group '{}' contains empty string".format(gn))
                for n in input_names:
                    input_desc = all_inputs.get(n.strip())
                    if input_desc is None:
                        raise RuntimeError("input '{}' of group '{}' not found".format(n, gn))
                    grouped_inputs[gn].append(input_desc)

        return grouped_inputs

    @classmethod
    def _parse_fcs(cls, fc_json, grouped_model_input_descs):
        if not fc_json:
            return None
        if not grouped_model_input_descs or not isinstance(grouped_model_input_descs, dict):
            raise RuntimeError("grouped_model_input_descs should be non-empty dict, got '{}': {}"
                               .format(type(grouped_model_input_descs), grouped_model_input_descs))

        if not isinstance(fc_json, dict):
            raise RuntimeError("'feature_columns' should be a dict, got '{}': {}".format(type(fc_json), fc_json))
        fc_defs = fc_json.get('defs', fc_json.get('defines'))
        if not fc_defs or not isinstance(fc_defs, (tuple, list)):
            raise RuntimeError("'defs/defines' in 'feature_columns' should be a non-empty tuple/list, got '{}': {}"
                               .format(type(fc_defs), fc_defs))

        grouped_model_input_descs = list(chain.from_iterable(grouped_model_input_descs.values()))
        model_input_desc_dict = {d.name: d for d in grouped_model_input_descs}

        def __gen_fc_name(desc, exists_fc_names, exists_model_input_names):
            if isinstance(desc.inputs, (tuple, list)):
                base_name = '_cross_'.join(desc.inputs)
            else:
                base_name = desc.inputs + '_' + desc.op.__name__
            i = 1
            while base_name in exists_fc_names or base_name in exists_model_input_names:
                base_name += '_' + str(i)
                i += 1
            return base_name

        fc_dict = {}
        root_fp_descs = []
        non_root_fp_descs = []
        for fc_def_json in fc_defs:
            fp_desc = FeatureProcessDesc.parse_from_json(fc_def_json)
            out_names = fp_desc.outputs
            if isinstance(out_names, str):
                if not out_names:
                    out_names = __gen_fc_name(fp_desc, fc_dict.keys(), model_input_desc_dict.keys())
                    print("'out' not set of fc op '{}', generated output name '{}' for it, desc={}"
                          .format(fp_desc.op, out_names, fp_desc))
                    fp_desc = fp_desc._replace(outputs=out_names)
                out_names = [out_names]

            for on in out_names:
                if on in fc_dict or on in model_input_desc_dict:
                    raise RuntimeError("'out' '{}' in fc op '{}' collide with other fc names or model input names,"
                                       " existed fc names={}, existed model input names={}"
                                       .format(on, fp_desc.op, fc_dict.keys(), model_input_desc_dict.keys()))
                fc_dict[on] = None
            in_names = fp_desc.inputs
            if isinstance(in_names, str):
                in_names = [in_names]
            if all(n in model_input_desc_dict for n in in_names):
                root_fp_descs.append(fp_desc)
            else:
                non_root_fp_descs.append(fp_desc)

        if not root_fp_descs:
            raise RuntimeError("found no fc op with inputs only from model inputs")

        processed_fps = 0
        while len(root_fp_descs) > 0:
            fp_desc = root_fp_descs.pop()
            try:
                fcs = fp_desc.to_tf_fcs(model_input_desc_dict, fc_dict)
                if isinstance(fcs, list):
                    for n, fc in zip(fp_desc.outputs, fcs):
                        fc_dict[n] = fc
                else:
                    fc_dict[fp_desc.outputs] = fcs
                processed_fps += 1
                print("processed fc op desc {}, generated feature columns: {}".format(fp_desc, fcs))

                non_root_fp_descs_cpy = []
                for child_fp_desc in non_root_fp_descs:
                    in_names = child_fp_desc.inputs
                    if isinstance(in_names, str):
                        in_names = [in_names]
                    if all([n in model_input_desc_dict or fc_dict.get(n) is not None for n in in_names]):
                        root_fp_descs.append(child_fp_desc)
                    else:
                        non_root_fp_descs_cpy.append(child_fp_desc)
                non_root_fp_descs.clear()
                non_root_fp_descs = non_root_fp_descs_cpy
            except Exception as e:
                print("run fc op error, fp_desc={}, model_input_desc_dict={}, fc_dict={}: {}\n{}"
                      .format(fp_desc, model_input_desc_dict, fc_dict, e, traceback.format_exc()))
                raise e

        if non_root_fp_descs:
            raise RuntimeError("detected circle in fc ops: {}".format(non_root_fp_descs))
        assert len(fc_defs) == processed_fps

        grouped_fcs = defaultdict(list)
        fc_groups = fc_json.get("groups")
        if not fc_groups:
            grouped_fcs[cls.DEF_FEATURE_GROUP] = list(fc_dict.values())
        else:
            if not isinstance(fc_groups, dict):
                raise RuntimeError("'groups' of 'feature_columns' should be a dict, got '{}': {}"
                                   .format(type(fc_groups), fc_groups))
            for gn, g_fc_names in fc_groups.items():
                if not isinstance(gn, str) or not gn.strip():
                    raise RuntimeError("key in 'groups' of 'feature_columns' should be non-empty strings, got '{}': {}"
                                       .format(type(gn), gn))
                gn = gn.strip()
                if not isinstance(g_fc_names, (tuple, list)) or not g_fc_names:
                    raise RuntimeError("value in 'groups' of 'feature_columns' should be non-empty tuple/lists, "
                                       "got '{}': {}".format(type(g_fc_names), g_fc_names))
                if not all([n.strip() for n in g_fc_names]):
                    raise RuntimeError("names in fc group '{}' contains empty string".format(gn))
                for fn in g_fc_names:
                    fc = fc_dict.get(fn.strip())
                    if fc is None:
                        raise RuntimeError("fc '{}' of group '{}' not found".format(fn, gn))
                    grouped_fcs[gn].append(fc)
        return grouped_fcs

    @property
    def all_inputs(self) -> List[InputDesc]:
        if not self._input_index:
            return []
        return list(self._input_index.values())

    @property
    def all_fcs(self) -> List[fc_lib.FeatureColumn]:
        if not self._fc_index:
            return []
        return list(self._fc_index.values())

    @property
    def kv_config_file(self) -> str:
        return self._kv_config_file

    def get_input_by_name(self, name) -> InputDesc:
        return self._input_index.get(name)

    def get_fc_by_name(self, name) -> fc_lib.FeatureColumn:
        return self._fc_index.get(name)

    def get_inputs_by_group(self, group) -> List[InputDesc]:
        inputs = []
        if isinstance(group, str) and group.strip():
            group = [group.strip()]
        elif isinstance(group, (list, tuple)):
            group = list(filter(lambda x: x, map(lambda x: x.strip(), group)))

        dedup = set()
        for g in group:
            g_ins = self._grouped_inputs.get(g, [])
            for i in g_ins:
                if i.name in dedup:
                    continue
                inputs.append(i)
                dedup.add(i.name)
        return inputs

    def get_fcs_by_group(self, group) -> List[fc_lib.FeatureColumn]:
        fcs = []
        if isinstance(group, str) and group.strip():
            group = [group.strip()]
        elif isinstance(group, (list, tuple)):
            group = list(filter(lambda x: x, map(lambda x: x.strip(), group)))
        for g in group:
            fcs.extend(self._grouped_fcs.get(g, []))

        return fcs

    def get_config(self):
        from tensorflow.python.feature_column import serialization
        ser_inputs = {}
        for gn, ins in self._grouped_inputs.items():
            ser_inputs[gn] = [i.get_config() for i in ins]
        ser_fcs = {}
        if self._grouped_fcs:
            for gn, fcs in self._grouped_fcs.items():
                ser_fcs[gn] = serialization.serialize_feature_columns(fcs)
        config = {
            "inputs": ser_inputs,
            "feature_columns": ser_fcs,
            "kv_config_file": self._kv_config_file
        }
        # print("{}: serialize config: {}".format(self.__class__.__name__, config))
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.feature_column import serialization
        deser_inputs = {}
        all_inputs = {}
        for gn, ser_ins in config['inputs'].items():
            deser_inputs[gn] = []
            for i in ser_ins:
                name = i['name']
                idesc = all_inputs.get(name)
                if idesc is None:
                    idesc = InputDesc.from_config(i)
                deser_inputs[gn].append(idesc)
                all_inputs[idesc.name] = idesc

        for i in all_inputs.values():
            if i.weight_col:
                all_inputs[i.weight_col].is_weight_col = True

        deser_fcs = {}
        if config.get('feature_columns'):
            for gn, ser_fcs in config['feature_columns'].items():
                deser_fcs[gn] = serialization.deserialize_feature_columns(ser_fcs, custom_objects=custom_objects)
        kv_config_file = config.get('kv_config_file')
        obj = cls(deser_inputs, deser_fcs, kv_config_file)
        # print("{}: deserialized: {}".format(cls.__name__, obj))
        return obj

    @classmethod
    def parse(cls, config_file, pack_path=None, export_path=None):
        if not os.path.isfile(config_file):
            raise RuntimeError("model input configure file '{}' not exists".format(config_file))

        with open(config_file, 'r') as f:
            config_json = json.load(f)
            if pack_path or export_path:
                config_json = recur_expand_param(config_json, export_path, pack_path)
            version = str(config_json.get('version', 1))
            inputs_json = config_json.get('inputs')
            if not inputs_json:
                raise RuntimeError("'inputs' not set in model input configure file '{}'".format(config_file))

            if version == '1':
                inputs = cls._parse_inputs(inputs_json)
            elif version == '2':
                inputs = cls._parse_inputs_v2(inputs_json)
            else:
                raise RuntimeError("unsurpported model input file version '{}'".format(version))

            fc_json = config_json.get('feature_columns')
            if fc_json:
                fcs = cls._parse_fcs(fc_json, inputs)
            else:
                fcs = None

            kv_config_file = config_json.get('kv_config_file')

        return cls(inputs, fcs, kv_config_file)
