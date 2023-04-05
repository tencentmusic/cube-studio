from abc import ABC
from job.pkgs.tf.extend_utils import is_using_mixed_precision

from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_layers import ModelInputLayer, DNNLayer, AttentionLayer, LocalActivationUnit


class DINModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, 
                 attn_units=[36,1], attn_act=['prelu'], attn_mode='SUM', attn_norm=True,
                 dnn_units=[200,80,2], dnn_act=['prelu','prelu'], 
                 dnn_dropout=None, dnn_use_bn=None, dnn_l1_reg=None, dnn_l2_reg=None, name='DIN'):
        super(DINModel, self).__init__(name=name)

        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True)
        
        # Attention层的核心使用的是LocalActivationUnit
        self.attention_layer = AttentionLayer(
            LocalActivationUnit(hidden_units=attn_units, activations=attn_act, mode=attn_mode, normalization=attn_norm)
        )

        # 如果最后的分类是二分类, 则将2改为1, 然后使用sigmoid激活, 如果是多分类, 则使用softmax激活
        dnn_units_cpy = dnn_units.copy()
        dnn_units_cpy[-1] = 1 if dnn_units_cpy[-1]<3 else dnn_units_cpy[-1]
        self.output_layer = DNNLayer(dnn_units_cpy, dnn_act, None, dnn_dropout, dnn_use_bn, dnn_l1_reg, dnn_l2_reg)

        self.model_input_config = model_input_config
        self.attn_units = attn_units
        self.attn_act = attn_act
        self.attn_mode = attn_mode
        self.attn_norm = attn_norm
        self.dnn_units = dnn_units
        self.dnn_act = dnn_act
        self.dnn_dropout = dnn_dropout
        self.dnn_use_bn = dnn_use_bn
        self.dnn_l1_reg = dnn_l1_reg
        self.dnn_l2_reg = dnn_l2_reg
        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # DIN的输入特征中必须包括item组, 这是序列特征
        item_embeddings = self.input_layer(inputs, groups='item')
        item_vals = concat_func(item_embeddings, flatten=False)
        item_embeddings = tf.stack(item_vals, axis=-1) # 支持多个item特征, 但是计算attention时并不是特征之间计算, 这里会将item的不同特征融合到一起
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)

        # DIN的输入特征中必须包括target组, 这是用于计算attention的特征, 也是一个item
        target_embeddings = self.input_layer(inputs, groups='target')
        target_embeddings = concat_func(target_embeddings, flatten=False)
        target_embeddings = tf.stack(target_embeddings, axis=-1)
        target_embeddings = tf.reduce_mean(target_embeddings, axis=-1)

        # query: target; key: item_seq; value: item_seq; hist_len
        # hist_len可以由输入的特征提供, 也可以通过input_layer计算得到
        if 'hist_len' in inputs.keys():
            hist_len = inputs['hist_len']
        else:
            hist_len = self.input_layer.compute_mask_histlen(inputs, self.get_item_histlen_name(), return_seqlist=True)
        
        # 计算attention, 需要的输入为 [query: target; key: item_seq; value: item_seq; hist_len]
        behavior_embeddings = self.attention_layer([target_embeddings, item_embeddings, item_embeddings, hist_len]) # [batch_size, 1/seq_len, dim]
        behavior_embeddings = tf.reduce_sum(behavior_embeddings, 1)

        # 如果还有额外的user侧特征, 可以进行拼接
        user_embeddings = None
        if self.model_input_config.get_inputs_by_group('user'):
            user_embeddings = self.input_layer(inputs, groups='user')
            user_vals = concat_func(user_embeddings)
            user_embeddings = tf.concat(user_vals, axis=-1)
        # 如果还有额外的context侧特征, 可以进行拼接
        context_embeddings = None
        if self.model_input_config.get_inputs_by_group('context'):
            context_embeddings = self.input_layer(inputs, groups='context')
            context_vals = concat_func(context_embeddings)
            context_embeddings = tf.concat(context_vals, axis=-1)
        
        dnn_inputs = tf.concat([behavior_embeddings, target_embeddings], -1)
        if user_embeddings is not None:
            dnn_inputs = tf.concat([user_embeddings, dnn_inputs], -1)
        if context_embeddings is not None:
            dnn_inputs = tf.concat([dnn_inputs, context_embeddings], -1)
        
        # 将所有拼接后的特征输入到最后的DNN中
        outputs = self.output_layer(dnn_inputs) # [batch_size, dim]
        # 如果最后的分类是二分类, 则将2改为1, 然后使用sigmoid激活, 如果是多分类, 则使用softmax激活
        if self.dnn_units[-1]<3:
            outputs = tf.nn.sigmoid(outputs)
        else:
            outputs = tf.nn.softmax(outputs) # [batch_size, dim]

        if self.mixed_precision and outputs.dtype != tf.float32:
            outputs = tf.cast(outputs, tf.float32, self.name+"_mp_output_cast2float32")

        return outputs

    def get_item_histlen_name(self):
        feats = self.model_input_config.get_inputs_by_group('item')
        name = feats[0].name
        for feat in feats:
            if feat.val_sep is not None:
                name = feat.name
                break
        return name

    def with_target_inputs(self, inputs):
        target_feats = self.model_input_config.get_inputs_by_group('target')
        if target_feats:
            for target_feat in target_feats:
                if target_feat.name not in inputs.keys():
                    return False
            return True
        return False

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'attn_units': self.attn_units, 
            'attn_act': self.attn_act, 
            'attn_mode': self.attn_mode,
            'attn_norm': self.attn_norm,
            'dnn_units': self.dnn_units, 
            'dnn_act': self.dnn_act, 
            'dnn_dropout': self.dnn_dropout, 
            'dnn_use_bn': self.dnn_use_bn, 
            'dnn_l1_reg': self.dnn_l1_reg, 
            'dnn_l2_reg': self.dnn_l2_reg,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_save_signatures(self):
        call_fn_specs = self.input_layer.get_tensor_specs()
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(call_fn_specs)
        }
        return sigs
