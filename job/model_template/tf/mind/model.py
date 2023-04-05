from abc import ABC
from job.pkgs.tf.extend_utils import is_using_mixed_precision

from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_layers import ModelInputLayer, CapsuleLayer, LabelAwareAttention, DNNLayer


class MINDModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, 
                 item_embedding_dim, seq_max_len, k_max, dynamic_k=False, pow_p=1, capsule_size=None,
                 dnn_hidden_widthes=[64,32], dnn_act='relu', 
                 dnn_dropout=None, dnn_use_bn=False, dnn_l1_reg=None, dnn_l2_reg=None,
                 name='MIND'):
        super(MINDModel, self).__init__(name=name)
        
        if not isinstance(dnn_hidden_widthes, (tuple, list)):
            raise RuntimeError("dnn_hidden_layers must be a tuple/list, got '{}': {}"
                               .format(type(dnn_hidden_widthes), dnn_hidden_widthes))
        
        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True) # item and target embedding_dim must be item_embedding_dim
        
        # MIND模型通过一个胶囊网络来提取用户多兴趣
        self.multi_interest_extractor_layer = CapsuleLayer(item_embedding_dim, capsule_size or item_embedding_dim, seq_max_len, k_max)
        
        # MIND模型最终的用户多兴趣是在胶囊网络的输出再经过DNN后得到的
        dnn_hidden_widthes_cpy = dnn_hidden_widthes[:]
        if dnn_hidden_widthes_cpy[-1] != item_embedding_dim:
            dnn_hidden_widthes_cpy.append(item_embedding_dim)
        self.H_layers = DNNLayer(dnn_hidden_widthes_cpy, dnn_act, None, dnn_dropout, 
                                 dnn_use_bn, dnn_l1_reg, dnn_l2_reg, name='H_layers')
        
        # MIND模型在训练过程中需要和target计算label_aware_attention, 用这个结果再去计算loss
        self.label_aware_attention = LabelAwareAttention(k_max, pow_p)
        
        self.model_input_config = model_input_config
        self.item_embedding_dim = item_embedding_dim
        self.seq_max_len = seq_max_len
        self.k_max = k_max
        self.dynamic_k = dynamic_k
        self.pow_p = pow_p
        self.capsule_size = capsule_size
        self.dnn_hidden_widthes = dnn_hidden_widthes
        self.dnn_act = dnn_act
        self.dnn_dropout = dnn_dropout
        self.dnn_use_bn = dnn_use_bn
        self.dnn_l1_reg = dnn_l1_reg
        self.dnn_l2_reg = dnn_l2_reg
        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # get user multi interest embeddings
        # 提取用户多兴趣, 这部分单独拆解出了一个函数
        user_interests_final, hist_len = self.get_user_multi_interests_inside(inputs) # [batch_size, k_max, item_embedding_dim]
        
        # get target embeddings
        target_embeddings = None
        if self.with_target_inputs(inputs):
            target_embeddings = self.input_layer(inputs, groups='target')
            target_embeddings = concat_func(target_embeddings, flatten=False)
            target_embeddings = tf.stack(target_embeddings, axis=-1)
            target_embeddings = tf.reduce_mean(target_embeddings, axis=-1)
            if len(target_embeddings.shape)==2:
                target_embeddings = tf.expand_dims(target_embeddings, -2) # [batch_size, 1, item_embedding_dim]
        
        if target_embeddings is not None:
            # calculate label aware attention
            # 如果包含target组特征, 说明这是训练过程, 需要计算label_aware_attention, 并将其作为模型输出进行loss计算
            # 为了配合多兴趣下的topk_hit_rate计算, 模型输出会是attention和用户多兴趣的拼接
            if self.dynamic_k and hist_len is not None:
                attns = self.label_aware_attention((user_inetrests_final, target_embeddings, hist_len)) # [batch_size, item_embedding_dim]
            else:
                attns = self.label_aware_attention((user_interests_final, target_embeddings)) # [batch_size, item_embedding_dim]
            return tf.concat([tf.expand_dims(attns, -2), user_interests_final], -2) # [batch_size, 1 + k_max, item_embedding_dim]
        else:
            # 如果不包含target组特征, 说明这是服务过程, 则输出用户多兴趣表征
            return user_interests_final # [batch_size, k_max, item_embedding_dim]
        
    # @tf.function
    def get_user_multi_interests_inside(self, inputs):
        user_embeddings = None
        if self.model_input_config.get_inputs_by_group('user'):
            user_embeddings = self.input_layer(inputs, groups='user')
        
        # 输入特征中必须包括item组的特征, 这是序列特征
        item_embeddings = self.input_layer(inputs, groups='item')
        item_vals = concat_func(item_embeddings, flatten=False)
        # pooling item features
        item_embeddings = tf.stack(item_vals, axis=-1, name='item_embeddings') # 支持多个item特征, 最后会将item的不同特征融合到一起
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)
        
        # hist_len可以由输入的特征提供, 也可以通过input_layer计算得到
        if 'hist_len' in inputs.keys():
            hist_len = inputs['hist_len']
        else:
            hist_len = self.input_layer.compute_mask_histlen(inputs, self.get_item_histlen_name(), return_seqlist=True)
        
        user_interests, hist_len = self.multi_interest_extractor_layer((item_embeddings, hist_len)) # [batch_size, k_max, item_embedding_dim]
        
        # 如果输入特征中包括了user组的特征, 将其和用户多兴趣拼接
        if user_embeddings is not None:
            user_vals = concat_func(user_embeddings)
            user_embeddings = tf.concat(user_vals, axis=-1) # [batch_size, feature_nums*embedding_dims]
            user_embeddings = tf.tile(tf.expand_dims(user_embeddings, -2), [1, self.k_max, 1])
            dnn_inputs = tf.concat([user_interests, user_embeddings], axis=-1) #[batch_size, k_max, item_embedding_dim + feature_nums*embedding_dims]
        else:
            dnn_inputs = user_interests
            
        # 将用户特征和提取到的用户多兴趣进行拼接后输入到DNN中, 得到最终的用户多兴趣表征
        user_interests_final = self.H_layers(dnn_inputs)
        
        # [batch_size, k_max, item_embedding_dim]
        return user_interests_final, hist_len

    @tf.function
    def get_user_multi_interests(self, inputs):
        user_interests_final, hist_len = self.get_user_multi_interests_inside(inputs)

        return user_interests_final

    def get_item_embeddings_model(self):
        # 负责Item Embedding导出的模型, 保存该模型用于新的使用predict_args方式来进行embedding导出
        inputs = {}
        # for i_desc in self.input_layer.get_feature_input_descs('item'):
        #     inputs[i_desc.name] = i_desc.to_tf_input()
        # item_embeddings = self.input_layer(inputs, groups='item')
        
        item_embeddings = {}
        for i_desc in self.get_item_labels():
            inputs[i_desc.name] = i_desc.to_tf_input()
            item_feature_embedding = self.get_item_feature_embeddings(inputs[i_desc.name], i_desc.embedding_name)
            item_embeddings[i_desc.name]  = item_feature_embedding

        item_vals = concat_func(item_embeddings, flatten=False)
        item_embeddings = tf.stack(item_vals, axis=-1, name='item_embeddings')
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)

        return tf.keras.Model(inputs, outputs=item_embeddings, name=self.name + "-item_embeddings_model")

    @tf.function
    def get_item_embeddings(self, item_inputs):
        # 服务于旧的类似YouTubeDNN的embedding导出
        # item_features_desc = self.model_input_config.get_inputs_by_group('item')
        item_embeddings = self.input_layer(item_inputs, groups='item')
        item_vals = concat_func(item_embeddings)
        item_embeddings = tf.stack(item_vals, axis=-1, name='item_embeddings')
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)
        
        return item_embeddings
    
    # @tf.function
    def get_item_feature_embeddings(self, item_list, item_feature_name):
        # 服务于旧的类似YouTubeDNN的embedding导出
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(item_feature_name) # must be item id
        item_list = tf.reshape(item_list, (-1, 1))
        item_embeddings = item_embedding_layer(item_list)
        item_embeddings = tf.reshape(item_embeddings, (-1, item_embedding_layer.embedding_dim))
        return item_embeddings
    
    def get_item_feature_embedding_weights(self, item_feature_name):
        # 服务于TopKHitRate
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(item_feature_name)
        return item_embedding_layer.get_embedding_matrix()
    
    def get_item_feature_cardinality(self, item_feature_name):
        # 服务于SampledSoftmax
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(item_feature_name)
        return item_embedding_layer.get_vocab_size()

    def lookup_item_index(self, item_list, item_feature_name):
        # 服务于SampledSoftmax
        item_embedding_layer = self.input_layer.get_embedding_layer_by_name(item_feature_name)
        return item_embedding_layer.word_to_index(item_list)    

    def get_item_label(self):
        # 获取表示label的特征中的第一个
        feats = [feat for feat in self.model_input_config.all_inputs if feat.is_label]
        return feats[0]

    def get_item_labels(self):
        # 获取表示label的特征
        return [feat for feat in self.model_input_config.all_inputs if feat.is_label]

    def get_item_histlen_name(self):
        # 获取序列特征的特征名
        feats = self.model_input_config.get_inputs_by_group('item')
        name = feats[0].name
        for feat in feats:
            if feat.val_sep is not None:
                name = feat.name
                break
        return name

    def with_target_inputs(self, inputs):
        # 判断输入中是否包含target组的特征, 从而判断是在训练过程还是服务过程
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
            'item_embedding_dim': self.item_embedding_dim,
            'seq_max_len': self.seq_max_len,
            'k_max': self.k_max,
            'dynamic_k': self.dynamic_k,
            'pow_p': self.pow_p,
            'capsule_size': self.capsule_size,
            'dnn_hidden_widthes': self.dnn_hidden_widthes,
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
