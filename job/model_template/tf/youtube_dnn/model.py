# coding=utf-8
# @Time     : 2021/1/6 17:12
# @Auther   : lionpeng@tencent.com
from abc import ABC

from job.pkgs.tf.extend_layers import ModelInputLayer, DNNLayer
from job.pkgs.tf.extend_utils import is_using_mixed_precision
from job.pkgs.tf.feature_util import *


class YoutubeDNNModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, dnn_hidden_widthes, dnn_hidden_active_fn=None,
                 dnn_output_active_fn=None, dnn_dropout_prob=None, dnn_use_bn=False, dnn_l1_reg=None,
                 dnn_l2_reg=None, name="youtubednn"):

        super(YoutubeDNNModel, self).__init__(name=name)
        if not model_input_config.get_inputs_by_group('user'):
            raise RuntimeError("'user' input group not exists in model input config")

        item_id_input_desc = model_input_config.get_inputs_by_group('item')
        if not item_id_input_desc or len(item_id_input_desc) != 1:
            raise RuntimeError("'item' input group must contain one and only one input about item_id, got {} from "
                               "model input config".format(item_id_input_desc))
        self.item_id_input_desc = item_id_input_desc[0]

        dnn_hidden_widthes = dnn_hidden_widthes or []
        if not isinstance(dnn_hidden_widthes, (tuple, list)):
            raise RuntimeError("dnn_hidden_layers must be a tuple/list, got '{}': {}"
                               .format(type(dnn_hidden_widthes), dnn_hidden_widthes))
        dnn_hidden_widthes.append(self.item_id_input_desc.embedding_dim)

        self.input_layer = ModelInputLayer(model_input_config, groups=['user', 'item'], auto_create_embedding=True)

        self.dnn = DNNLayer(dnn_hidden_widthes, dnn_hidden_active_fn, dnn_output_active_fn, dnn_dropout_prob,
                            dnn_use_bn, dnn_l1_reg, dnn_l2_reg, name=name+"/dnn")

        self.model_input_config = model_input_config
        self.dnn_hidden_widthes = dnn_hidden_widthes
        self.dnn_hidden_active_fn = dnn_hidden_active_fn
        self.dnn_output_active_fn = dnn_output_active_fn
        self.dnn_dropout_prob = dnn_dropout_prob
        self.dnn_use_bn = dnn_use_bn
        self.dnn_l1_reg = dnn_l1_reg
        self.dnn_l2_reg = dnn_l2_reg
        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        transformed_inputs = self.input_layer(inputs)
        feat_vals = []
        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_user_features')
        user_embeddings = self.dnn(concat_vals, training=training)
        if self.mixed_precision and user_embeddings.dtype != tf.float32:
            user_embeddings = tf.cast(user_embeddings, tf.float32, self.name+"_useremb_cast2float32")
        return user_embeddings

    @tf.function
    def predict_item_embeddings(self, item_ids):
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(self.item_id_input_desc.embedding_name)
        item_ids = tf.reshape(item_ids, (-1, 1))
        item_embeddings = item_embedding_layer(item_ids)
        item_embeddings = tf.reshape(item_embeddings, (-1, item_embedding_layer.embedding_dim))
        return item_embeddings

    def get_item_embedding_weights(self):
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(self.item_id_input_desc.embedding_name)
        return item_embedding_layer.get_embedding_matrix()

    def get_item_cardinality(self):
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(self.item_id_input_desc.embedding_name)
        return item_embedding_layer.get_vocab_size()

    def lookup_item_index(self, item_ids):
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(self.item_id_input_desc.embedding_name)
        return item_embedding_layer.word_to_index(item_ids)

    def get_save_signatures(self):
        call_fn_specs = self.input_layer.get_tensor_specs()
        pred_item_emb_fn_spec = self.item_id_input_desc.to_tf_tensor_spec()
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(call_fn_specs),
            "predict_item_embeddings": self.predict_item_embeddings.get_concrete_function(pred_item_emb_fn_spec)
        }
        return sigs

    def get_config(self):
        dnn_hidden_widthes = self.dnn_hidden_widthes.copy()
        dnn_hidden_widthes.pop(-1)
        config = {
            'model_input_config': self.model_input_config,
            'dnn_hidden_widthes': dnn_hidden_widthes,
            'dnn_hidden_active_fn': self.dnn_hidden_active_fn,
            'dnn_output_active_fn': self.dnn_output_active_fn,
            'dnn_dropout_prob': self.dnn_dropout_prob,
            'dnn_use_bn': self.dnn_use_bn,
            'dnn_l1_reg': self.dnn_l1_reg,
            'dnn_l2_reg': self.dnn_l2_reg,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
