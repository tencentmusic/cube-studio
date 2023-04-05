# coding=utf-8
# @Time     : 2021/5/27 15:46
# @Auther   : lionpeng@tencent.com
from abc import ABC

from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_layers import ModelInputLayer, DNNLayer


class ESMMModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, cvr_dnn_units=[1], cvr_dnn_hidden_act=None,
                 cvr_dnn_use_bias=True, cvr_dnn_use_bn=False, cvr_dnn_dropout=None,
                 cvr_dnn_l1_reg=None, cvr_dnn_l2_reg=None, ctr_dnn_units=[1], ctr_dnn_hidden_act=None,
                 ctr_dnn_use_bias=True, ctr_dnn_use_bn=False, ctr_dnn_dropout=None,
                 ctr_dnn_l1_reg=None, ctr_dnn_l2_reg=None, ctr_label_name='', ctcvr_label_name='',
                 name="esmm"):
        super(ESMMModel, self).__init__(name=name)

        if not isinstance(cvr_dnn_units, (list, tuple)) or not cvr_dnn_units or \
                not all([isinstance(i, int) and i > 0 for i in cvr_dnn_units]):
            raise RuntimeError("'cvr_dnn_units' should be a non-empty list of positive integers, got '{}': {}"
                               .format(type(cvr_dnn_units), cvr_dnn_units))

        if cvr_dnn_units[-1] != 1:
            cvr_dnn_units.append(1)
            print("{}: WARNING: last dimemsion of 'cvr_dnn_units' is not 1, auto append 1: {}"
                  .format(self.name, cvr_dnn_units))

        if not isinstance(ctr_dnn_units, (list, tuple)) or not ctr_dnn_units or \
                not all([isinstance(i, int) and i > 0 for i in ctr_dnn_units]):
            raise RuntimeError("'cvr_dnn_units' should be a non-empty list of positive integers, got '{}': {}"
                               .format(type(ctr_dnn_units), ctr_dnn_units))

        if ctr_dnn_units[-1] != 1:
            ctr_dnn_units.append(1)
            print("{}: WARNING: last dimemsion of 'cvr_dnn_units' is not 1, auto append 1: {}"
                  .format(self.name, ctr_dnn_units))

        if not isinstance(ctr_label_name, str) or not ctr_label_name.strip():
            raise RuntimeError("'ctr_label_name' should be a non-empty string, got '{}': {}"
                               .format(type(ctr_label_name), ctr_label_name))
        ctr_label_name = ctr_label_name.strip()

        if model_input_config.get_input_by_name(ctr_label_name) is None:
            raise RuntimeError("'ctr_label_name' '{}' not found in model inputs".format(ctr_label_name))

        if not isinstance(ctcvr_label_name, str) or not ctcvr_label_name.strip():
            raise RuntimeError("'ctcvr_label_name' should be a non-empty string, got '{}': {}"
                               .format(type(ctcvr_label_name), ctcvr_label_name))
        ctcvr_label_name = ctcvr_label_name.strip()

        if model_input_config.get_input_by_name(ctcvr_label_name) is None:
            raise RuntimeError("'ctcvr_label_name' '{}' not found in model inputs".format(ctcvr_label_name))

        if ctr_label_name == ctcvr_label_name:
            raise RuntimeError("'ctr_label_name' and 'ctcvr_label_name' can not be the same: '{}'"
                               .format(ctr_label_name))

        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True,
                                           name=self.name+"/input_layer")

        self.cvr_dnn = DNNLayer(cvr_dnn_units, cvr_dnn_hidden_act, "sigmoid", cvr_dnn_dropout,
                                cvr_dnn_use_bn, cvr_dnn_l1_reg, cvr_dnn_l2_reg, cvr_dnn_use_bias,
                                name=self.name+"/cvr_dnn_layer")

        self.ctr_dnn = DNNLayer(ctr_dnn_units, ctr_dnn_hidden_act, "sigmoid", ctr_dnn_dropout,
                                ctr_dnn_use_bn, ctr_dnn_l1_reg, ctr_dnn_l2_reg, ctr_dnn_use_bias,
                                name=self.name+"/ctr_dnn_layer")

        self.model_input_config = model_input_config
        self.cvr_dnn_units = cvr_dnn_units
        self.cvr_dnn_hidden_act = cvr_dnn_hidden_act
        self.cvr_dnn_use_bias = cvr_dnn_use_bias
        self.cvr_dnn_use_bn = cvr_dnn_use_bn
        self.cvr_dnn_dropout = cvr_dnn_dropout
        self.cvr_dnn_l1_reg = cvr_dnn_l1_reg
        self.cvr_dnn_l2_reg = cvr_dnn_l2_reg
        self.ctr_dnn_units = ctr_dnn_units
        self.ctr_dnn_hidden_act = ctr_dnn_hidden_act
        self.ctr_dnn_use_bias = ctr_dnn_use_bias
        self.ctr_dnn_use_bn = ctr_dnn_use_bn
        self.ctr_dnn_dropout = ctr_dnn_dropout
        self.ctr_dnn_l1_reg = ctr_dnn_l1_reg
        self.ctr_dnn_l2_reg = ctr_dnn_l2_reg
        self.ctr_label_name = ctr_label_name
        self.ctcvr_label_name = ctcvr_label_name

    @tf.function
    def call(self, inputs, training=None, mask=None):
        cvr_feat_vals = []
        transformed_cvr_inputs = self.input_layer(inputs, groups='cvr')
        for val in transformed_cvr_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            cvr_feat_vals.append(val)
        cvr_concat_vals = tf.concat(cvr_feat_vals, axis=-1, name='concat_cvr_features')
        cvr_pred = self.cvr_dnn(cvr_concat_vals, training=training)

        ctr_feat_vals = []
        transformed_ctr_inputs = self.input_layer(inputs, groups='ctr')
        for val in transformed_ctr_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            ctr_feat_vals.append(val)
        ctr_concat_vals = tf.concat(ctr_feat_vals, axis=-1, name='concat_ctr_features')
        ctr_pred = self.ctr_dnn(ctr_concat_vals, training=training)

        ctcvr_pred = ctr_pred * cvr_pred
        pred = {self.ctr_label_name: ctr_pred, self.ctcvr_label_name: ctcvr_pred}
        return pred

    def get_call_signatures(self):
        fn_specs = {}
        cvr_inputs_specs = self.input_layer.get_tensor_specs('cvr')
        fn_specs.update(cvr_inputs_specs)
        ctr_inputs_specs = self.input_layer.get_tensor_specs('ctr')
        fn_specs.update(ctr_inputs_specs)

        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(fn_specs),
        }
        return sigs

    def get_cvr_model(self):
        inputs = {}
        groups = 'cvr'
        for i_desc in self.input_layer.get_feature_input_descs(groups):
            inputs[i_desc.name] = i_desc.to_tf_input()
        feat_vals = []
        transformed_inputs = self.input_layer(inputs, groups=groups)
        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_cvr_features')
        cvr_pred = self.cvr_dnn(concat_vals)
        return tf.keras.Model(inputs, outputs=cvr_pred, name=self.name + "-cvr")

    def get_ctr_model(self):
        inputs = {}
        groups = 'ctr'
        for i_desc in self.input_layer.get_feature_input_descs(groups):
            inputs[i_desc.name] = i_desc.to_tf_input()
        feat_vals = []
        transformed_inputs = self.input_layer(inputs, groups=groups)
        for val in transformed_inputs.values():
            val = tf.keras.layers.Flatten()(val)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_cvr_features')
        ctr_pred = self.ctr_dnn(concat_vals)
        return tf.keras.Model(inputs, outputs=ctr_pred, name=self.name + "-ctr")

    def get_config(self):
        config = {
            "model_input_config": self.model_input_config,
            "cvr_dnn_units": self.cvr_dnn_units,
            "cvr_dnn_hidden_act": self.cvr_dnn_hidden_act,
            "cvr_dnn_use_bias": self.cvr_dnn_use_bias,
            "cvr_dnn_use_bn": self.cvr_dnn_use_bn,
            "cvr_dnn_dropout": self.cvr_dnn_dropout,
            "cvr_dnn_l1_reg": self.cvr_dnn_l1_reg,
            "cvr_dnn_l2_reg": self.cvr_dnn_l2_reg,
            "ctr_dnn_units": self.ctr_dnn_units,
            "ctr_dnn_hidden_act": self.ctr_dnn_hidden_act,
            "ctr_dnn_use_bias": self.ctr_dnn_use_bias,
            "ctr_dnn_use_bn": self.ctr_dnn_use_bn,
            "ctr_dnn_dropout": self.ctr_dnn_dropout,
            "ctr_dnn_l1_reg": self.ctr_dnn_l1_reg,
            "ctr_dnn_l2_reg": self.ctr_dnn_l2_reg,
            "ctr_label_name": self.ctr_label_name,
            "ctcvr_label_name": self.ctcvr_label_name,
            "name": self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
