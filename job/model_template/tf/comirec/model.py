from abc import ABC

from job.pkgs.tf.extend_layers import (CapsuleLayer, DNNLayer, ModelInputLayer,
                                       PositionalEncodingLayer)
from job.pkgs.tf.extend_utils import is_using_mixed_precision
from job.pkgs.tf.feature_util import *


class ComiRecModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, 
                 item_embedding_dim, seq_max_len, interest_extractor='DR', 
                 num_interests=3, hidden_size=None, capsule_size=None, add_pos=None, pow_p=1,
                 name='ComiRec'):
        super(ComiRecModel, self).__init__(name=name)
        
        assert interest_extractor.lower() in ['dr','sa']
        
        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True) # item and target embedding_dim must be item_embedding_dim
        
        if interest_extractor.lower()=='dr':
            self.capsule_layer = CapsuleLayer(item_embedding_dim, item_embedding_dim, seq_max_len, num_interests)
        elif interest_extractor.lower()=='sa':
            if add_pos:
                self.positional_encoding_layer = PositionalEncodingLayer(max_len=seq_max_len, dim=item_embedding_dim, 
                                                                         learnable=True)
            self.attention_layer = DNNLayer(
                [hidden_size or item_embedding_dim*4, num_interests],
                ['tanh', None], name='Self-Attn-DNN')
        
        self.model_input_config = model_input_config
        self.interest_extractor = interest_extractor.lower()
        self.item_embedding_dim = item_embedding_dim
        self.seq_max_len = seq_max_len
        self.num_interests = num_interests
        self.capsule_size = capsule_size
        self.hidden_size = hidden_size
        self.add_pos = add_pos
        self.pow_p = pow_p
        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    def _compute_user_interest(self, inputs):
        # 将用户多兴趣提取部分单独分离出来
        # 输入特征中必须包括item组的特征, 这是序列特征
        item_embeddings = self.input_layer(inputs, groups='item')
        item_vals = concat_func(item_embeddings, flatten=False)
        item_embeddings = tf.stack(item_vals, axis=-1)
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)

        # hist_len可以由输入的特征提供, 也可以通过input_layer计算得到, 如果是服务阶段, 由于模型加载的原因, 需要自己提供hist_len特征
        if 'hist_len' in inputs.keys():
            hist_len = inputs['hist_len']
        else:
            hist_len = self.input_layer.compute_mask_histlen(inputs, self.get_item_histlen_name(), return_seqlist=True)

        interests = None
        if self.interest_extractor == 'dr':
            # 如果用动态路由算法, 则通过胶囊网络来提取多兴趣
            interests, hist_len = self.capsule_layer((item_embeddings, hist_len))  # [batch_size, num_interests, dim]
        elif self.interest_extractor == 'sa':
            # 如果用自注意力算法, 则通过DNN来提取多兴趣(论文中使用DNN而不是类似Transformer那样的QK点积计算)
            if self.add_pos:
                # 如果需要添加位置编码, 则根据论文所述添加可学习的位置编码
                position_embedding = self.positional_encoding_layer(inputs[self.get_item_histlen_name()])
                item_embeddings = item_embeddings + position_embedding
            # get attention
            attn = self.attention_layer(item_embeddings)  # [batch_size, seq_len, num_interests]

            # padding and mask
            seq_len_tile = tf.tile(hist_len, [1, self.num_interests])  # [batch_size, num_interests]
            mask = tf.sequence_mask(seq_len_tile, self.seq_max_len)  # [batch_size, num_interests, max_len]
            mask = tf.transpose(mask, [0, 2, 1])  # [batch_size, max_len, num_interests]
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [batch_size, max_len, num_interests]

            attn = tf.where(mask, attn, pad)  # [batch_size, seq_len, num_interests]
            attn = tf.nn.softmax(attn, -2)  # [batch_size, seq_len, num_interests] # softmax on dim seq_len
            attn = tf.transpose(attn, [0, 2, 1])  # [batch_size, num_interests, seq_len]

            interests = tf.matmul(attn, item_embeddings)  # [B, N_I, L] @ [B, L, D] -> [batch_size, num_interests, dim]
        return interests, item_embeddings

    @tf.function
    def call(self, inputs, training=None, mask=None):
        # 获取用户多兴趣
        interests, item_embeddings = self._compute_user_interest(inputs)

        if self.with_target_inputs(inputs):
            # 如果包含target组特征, 说明这是训练过程, 需要计算attention, 并将其作为模型输出进行loss计算
            # 为了配合多兴趣下的topk_hit_rate计算, 模型输出会是attention和用户多兴趣的拼接
            target_embeddings = self.input_layer(inputs, groups='target')
            target_embeddings = concat_func(target_embeddings, flatten=False)
            target_embeddings = tf.stack(target_embeddings, axis=-1)
            target_embeddings = tf.reduce_mean(target_embeddings, axis=-1) # [batch_size, item_embedding_dim]

            attn_n = tf.matmul(interests, tf.expand_dims(target_embeddings, -1)) # [B, N_I, D]@[B, D, 1] -> [batch_size, num_interests, 1]
            attn_n = tf.nn.softmax(tf.pow(tf.squeeze(attn_n, -1), self.pow_p)) # [batch_size, num_interests]

            out = tf.gather(
                tf.reshape(interests, [-1, self.item_embedding_dim]),
                tf.argmax(attn_n, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_embeddings)[0])*self.num_interests
            ) # [batch_size, item_embedding_dim]
            
            return out
            # return tf.concat([tf.expand_dims(out, -2), interests], -2) # [batch_size, 1 + num_interests, item_embedding_dim]
            # return [out, interests] # [batch_size, item_embedding_dim], [batch_size, num_iterests, item_embedding_dim]
        else:
            return interests # [batch_size, num_iterests, item_embedding_dim]

    def get_user_interests_model(self):
         # 负责用户多兴趣导出的模型, 保存该模型用于服务阶段获取用户多兴趣
        inputs = {}
        for i_desc in self.input_layer.get_feature_input_descs(['item', 'item_len']):
            inputs[i_desc.name] = i_desc.to_tf_input()

        # [batch_size, num_interests, dim],
        interests, _ = self._compute_user_interest(inputs)
        # [batch_size, num_interests*dim]
        flatten_interests = tf.keras.layers.Flatten()(interests)
        return tf.keras.Model(inputs, outputs=flatten_interests, name=self.name + "-user_model")

    def get_item_embeddings_model(self):
        # 负责Item Embedding导出的模型, 保存该模型用于新的使用predict_args方式来进行embedding导出
        inputs = {}
        item_embeddings = {}
        for i_desc in self.get_item_labels():
            inputs[i_desc.name] = i_desc.to_tf_input()
            item_feature_embedding = self.get_item_feature_embeddings(inputs[i_desc.name], i_desc.embedding_name)
            item_embeddings[i_desc.name] = item_feature_embedding

        item_vals = concat_func(item_embeddings, flatten=False)
        item_embeddings = tf.stack(item_vals, axis=-1)
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)

        return tf.keras.Model(inputs, outputs=item_embeddings, name=self.name + "-item_embeddings_model")

    def get_item_embeddings(self, item_inputs):
        # 服务于旧的类似YouTubeDNN的embedding导出
        item_embeddings = self.input_layer(item_inputs, groups='item')
        item_vals = concat_func(item_embeddings)
        item_embeddings = tf.stack(item_vals, axis=-1)
        item_embeddings = tf.reduce_mean(item_embeddings, axis=-1)
        
        return item_embeddings
    
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
    
    def get_target_inputs(self, inputs):
        # 获取target组的特征
        target_feats = self.model_input_config.get_inputs_by_group('target')
        if target_feats:
            res = [target_feat.name for target_feat in target_feats]
            return res
        return []

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'interest_extractor': self.interest_extractor,
            'item_embedding_dim': self.item_embedding_dim,
            'seq_max_len': self.seq_max_len,
            'num_interests': self.num_interests,
            'capsule_size': self.capsule_size,
            'hidden_size': self.hidden_size,
            'add_pos': self.add_pos,
            'pow_p': self.pow_p,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_save_signatures(self):
        call_fn_specs = self.input_layer.get_tensor_specs("item")
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(call_fn_specs)
        }
        return sigs
