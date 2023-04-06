# coding=utf-8
# @Time     : 2021/2/1 11:49
# @Auther   : lionpeng@tencent.com
from abc import ABC

from job.pkgs.tf.extend_layers import DNNLayer, FMLayer, ModelInputLayer
from job.pkgs.tf.extend_utils import is_using_mixed_precision
from job.pkgs.tf.feature_util import *


class DeepFMModel(tf.keras.models.Model, ABC):

    def __init__(self, model_input_config: ModelInputConfig, k, dnn_hidden_widthes, dnn_hidden_active_fn=None,
                 dnn_dropout_prob=None, dnn_use_bn=False, dnn_l1_reg=None, dnn_l2_reg=None, embedding_l1_reg=None,
                 embedding_l2_reg=None, name="deepfm"):
        super(DeepFMModel, self).__init__(name=name)

        if not isinstance(k, int) or k <= 0:
            raise RuntimeError("'k' should be a positive int, got '{}': {}".format(type(k), k))

        self.input_layer = ModelInputLayer(model_input_config, name=self.name+'_input_layer')
        self.fm = FMLayer.create_from_model_input_config(model_input_config, embedding_dim=k, with_logits=True,
                                                         use_bias=False, embedding_l1_reg=embedding_l1_reg,
                                                         embedding_l2_reg=embedding_l2_reg)

        if dnn_hidden_widthes:
            dnn_hidden_widthes_cpy = dnn_hidden_widthes[:]
            if dnn_hidden_widthes_cpy[-1] != 1:
                dnn_hidden_widthes_cpy.append(1)
                print("append last dnn output layer: {}".format(dnn_hidden_widthes_cpy))
            else:
                print("user specified dnn output layer: {}".format(dnn_hidden_widthes_cpy))
            self.dnn = DNNLayer(dnn_hidden_widthes_cpy, dnn_hidden_active_fn, None, dnn_dropout_prob,
                                dnn_use_bn, dnn_l1_reg, dnn_l2_reg, name="dnn")
        else:
            print("'dnn_hidden_widthes' not specified, will not use dnn")
            self.dnn = None

        self.model_input_config = model_input_config
        self.k = k
        self.dnn_hidden_widthes = dnn_hidden_widthes
        self.dnn_hidden_active_fn = dnn_hidden_active_fn
        self.dnn_dropout_prob = dnn_dropout_prob
        self.dnn_use_bn = dnn_use_bn
        self.dnn_l1_reg = dnn_l1_reg
        self.dnn_l2_reg = dnn_l2_reg
        self.embedding_l1_reg = embedding_l1_reg
        self.embedding_l2_reg = embedding_l2_reg

        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        inputs = self.input_layer(inputs)
        fm_logits, fm_latent_matrix = self.fm(inputs)
        if self.dnn is not None:
            # [batch, #fields * embedding_dim]
            fm_lantent_matrix = tf.keras.layers.Flatten()(fm_latent_matrix)
            # [batch, 1]
            dnn_logits = self.dnn(fm_lantent_matrix, training=training)
            # [batch, 1]
            logits = fm_logits + dnn_logits
        else:
            # [batch, 1]
            logits = fm_logits
        out = tf.nn.sigmoid(logits, name=self.name+'_output')
        if self.mixed_precision and out.dtype != tf.float32:
            out = tf.cast(out, tf.float32, self.name+"_mp_output_cast2float32")
        return out

    def get_config(self):
        config = {
            "model_input_config": self.model_input_config,
            "k": self.k,
            "dnn_hidden_widthes": self.dnn_hidden_widthes,
            "dnn_hidden_active_fn": self.dnn_hidden_active_fn,
            "dnn_dropout_prob": self.dnn_dropout_prob,
            "dnn_use_bn": self.dnn_use_bn,
            "dnn_l1_reg": self.dnn_l1_reg,
            "dnn_l2_reg": self.dnn_l2_reg,
            "embedding_l1_reg": self.embedding_l1_reg,
            "embedding_l2_reg": self.embedding_l2_reg,
            "name": self.name
        }
        # print("{}: serialize config: {}".format(self.__class__.__name__, config))
        return config

    @classmethod
    def from_config(cls, config):
        m = cls(**config)
        # print("{}: deserialized: {}".format(cls.__name__, m))
        return m

    def get_save_signatures(self):
        call_fn_specs = self.input_layer.get_tensor_specs()
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.function(self.call).get_concrete_function(call_fn_specs)
        }
        return sigs

    def get_input_placeholders(self):
        in_phs = {}
        for fdesc in self.input_layer.get_feature_input_descs():
            in_phs[fdesc.name] = fdesc.to_tf1_placeholder()
        return in_phs
