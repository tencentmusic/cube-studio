from abc import ABC
from job.pkgs.tf.extend_utils import is_using_mixed_precision

from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_layers import ModelInputLayer, MMoELayer, DNNLayer, FMLayer
from job.pkgs.tf.helperfuncs import create_activate_func


class MMoEModelV2(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, task_structs, num_experts, expert_layers, 
                 use_wide=True, wide_type='FM', wide_width=None, feature_cross=None, output_cross=None,
                 task_use_bias=True, task_hidden_act=None, task_output_act=None, task_use_bn=False,
                 task_dropout=None, task_l1_reg=None, task_l2_reg=None, expert_use_bias=True, expert_act='relu',
                 expert_dropout=None, expert_use_bn=False, expert_l1_reg=None, expert_l2_reg=None, gate_use_bias=True,
                 gate_l1_reg=None, gate_l2_reg=None, share_gates=False, wide_l1_reg=None, wide_l2_reg=None, 
                 named_outputs=None, name='MMoEV2'):
        super(MMoEModelV2, self).__init__(name=name)

        if not isinstance(task_structs, (list, tuple)) or not task_structs or \
                not all([isinstance(s, (list, tuple)) for s in task_structs]):
            raise RuntimeError("'task_structs' should be a non-empty list of list, got '{}': {}"
                               .format(type(task_structs), task_structs))

        for i, ts in enumerate(task_structs):
            if not ts or not all([isinstance(i, int) and i > 0 for i in ts]):
                raise RuntimeError("{}th task struct is invalid: {}".format(i, ts))

        self.num_tasks = len(task_structs)

        def __normalize_task_dnn_args(args, args_name, arg_types):
            if isinstance(args, (list, tuple)):
                # if len(args) != self.num_tasks:
                #     raise RuntimeError("#{} != #task {}".format(args_name, len(args), self.num_tasks))
                if not all([isinstance(a, arg_types) for a in args]):
                    raise RuntimeError("'{}' should be list of {}, got: {}".format(args_name, arg_types, args))
                args = [a.strip() if isinstance(a, str) else a for a in args]
                args = [None if isinstance(i, str) and (not i or i.lower() == 'none') else i for i in args]
            elif isinstance(args, arg_types) or args is None:
                if isinstance(args, str):
                    args = args.strip()
                    args = None if not args or args.lower() == 'none' else args
                args = [args] * self.num_tasks
            else:
                raise RuntimeError("'{}' should be a {}/list of {}, got '{}': {}"
                                   .format(args_name, arg_types, arg_types, type(args), args))

            print("processed {}={}".format(args_name, args))
            return args

        task_use_bias_chk = __normalize_task_dnn_args(task_use_bias, 'task_use_bias', (bool, list))
        task_hidden_act_chk = __normalize_task_dnn_args(task_hidden_act, 'task_hidden_act', (str, list))
        task_output_act_chk = __normalize_task_dnn_args(task_output_act, 'task_output_act', str)
        task_use_bn_chk = __normalize_task_dnn_args(task_use_bn, 'task_use_bn', (bool, list))
        task_dropout_chk = __normalize_task_dnn_args(task_dropout, 'task_dropout', (float, list))
        task_l1_reg_chk = __normalize_task_dnn_args(task_l1_reg, 'task_l1_reg', float)
        task_l2_reg_chk = __normalize_task_dnn_args(task_l2_reg, 'task_l2_reg', float)

        self.task_towers = []
        self.task_active_layers = []
        idx = 0
        for struct, use_bias, hidden_act, output_act, use_bn, dropout, l1_reg, l2_reg \
                in zip(task_structs, task_use_bias_chk, task_hidden_act_chk, task_output_act_chk, task_use_bn_chk,
                       task_dropout_chk, task_l1_reg_chk, task_l2_reg_chk):
            # task tower
            self.task_towers.append(DNNLayer(struct, hidden_act, None, dropout, use_bn, l1_reg, l2_reg,
                                             use_bias, name='task_{}'.format(idx))) # output_active_fn=None
            # task activation
            output_act = output_act.strip() if isinstance(output_act, str) else None
            self.task_active_layers.append(
                tf.keras.layers.Activation(
                    create_activate_func(output_act), 
                    name=self.name+"/output_act_{}_task{}".format(output_act, idx)
                ))
            idx += 1

        self.mmoe_layer = MMoELayer(self.num_tasks, num_experts, expert_layers, expert_use_bias, expert_act,
                                    expert_dropout, expert_use_bn, expert_l1_reg, expert_l2_reg,
                                    gate_use_bias, gate_l1_reg, gate_l2_reg, share_gates)

        if use_wide:
            if wide_type=='FM':
                # 如果设置使用wide侧, 且wide侧为FM模型, 则会单独配置FM的input_layer
                self.wide_layer = FMLayer.create_from_model_input_config(model_input_config, groups='wide',
                                                         embedding_dim=wide_width, with_logits=False,
                                                         use_bias=False, embedding_l1_reg=wide_l1_reg, embedding_l2_reg=wide_l2_reg) # output_active_fn=None
                # if using FM, wide model FM need individual wide input layer
                self.wide_input_layer = ModelInputLayer(model_input_config, groups='wide', auto_create_embedding=False, name=self.name+'wide_input_layer')
            elif wide_type=='LR':
                # 如果设置使用wide侧, 且wide侧为LR模型, 则和Deep侧共享input_layer
                self.wide_layer = DNNLayer([1], 'relu', None, 0., True, wide_l1_reg, wide_l2_reg) # output_active_fn=None
            else:
                self.wide_layer = None
        
        if use_wide and wide_type=='FM':
            # if using FM, input layer of mmoe will use mmoe group features
            self.input_layer = ModelInputLayer(model_input_config, groups='mmoe', auto_create_embedding=True, name=self.name+'input_layer')
        else:
            # if not using FM, mmoe and LR will share input layer and features
            self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True, name=self.name+'input_layer')

        self.label_names = [i.name for i in model_input_config.all_inputs if i.is_label]
        print("label_names={}".format(self.label_names))

        if named_outputs is None:
            named_outputs = len(self.label_names) > 1
        print("named_outputs={}".format(named_outputs))

        self.model_input_config = model_input_config
        self.task_structs = task_structs
        self.num_experts = num_experts
        self.expert_layers = expert_layers
        self.task_use_bias = task_use_bias
        self.task_hidden_act = task_hidden_act
        self.task_output_act = task_output_act
        self.task_use_bn = task_use_bn
        self.task_dropout = task_dropout
        self.task_l1_reg = task_l1_reg
        self.task_l2_reg = task_l2_reg
        self.expert_use_bias = expert_use_bias
        self.expert_act = expert_act
        self.expert_dropout = expert_dropout
        self.expert_use_bn = expert_use_bn
        self.expert_l1_reg = expert_l1_reg
        self.expert_l2_reg = expert_l2_reg
        self.gate_use_bias = gate_use_bias
        self.gate_l1_reg = gate_l1_reg
        self.gate_l2_reg = gate_l2_reg
        self.wide_l1_reg = wide_l1_reg
        self.wide_l2_reg = wide_l2_reg
        self.use_wide = use_wide
        self.wide_type = wide_type
        self.wide_width = wide_width
        self.feature_cross = feature_cross
        self.output_cross = output_cross
        self.share_gates = share_gates
        self.named_outputs = named_outputs
        self.mixed_precision = is_using_mixed_precision(self)
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    @tf.function
    def call(self, inputs, training=None, mask=None):   
        # mmoe part
        if len(self.input_layer.get_input_config().get_inputs_by_group("mmoe"))>0:
            # 如果使用了wide侧模型, 那么输入特征中应该将deep侧特征组命名为mmoe
            mmoe_inputs = self.input_layer(inputs, groups='mmoe')
        else:
            mmoe_inputs = self.input_layer(inputs)

        feat_vals = []
        for name, val in mmoe_inputs.items():
            val = tf.keras.layers.Flatten()(val)
            if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                   .format(name, val.dtype))
            elif val.dtype != self._dtype_policy.compute_dtype:
                val = tf.cast(val, self._dtype_policy.compute_dtype,
                              name=name + '_cast2' + self._dtype_policy.compute_dtype)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_features')

        # wide part
        wide_logits = None
        if self.use_wide:
            # wide_logtis [batch, 1]    
            if self.wide_type == 'FM':
                # 如果使用FM作为wide侧模型, 则wide侧特征表示会使用FM得到的隐向量
                wide_inputs = self.wide_input_layer(inputs, groups='wide')
                wide_logits, wide_latent_matrix = self.wide_layer(wide_inputs, groups='wide')
                wide_latent_matrix = tf.keras.layers.Flatten()(wide_latent_matrix)
            else:
                # 如果使用LR作为wide侧模型, 则wide侧特征通过共享的input_layer得到
                # 这里是因为ModelInputLayer可以分组进行计算, 只需要一个即可, 而wide侧的组可以和mmoe侧的组拥有相同和不同的特征
                wide_inputs = self.input_layer(inputs, groups='wide')
                vec_inputs = []; cross_inputs = []
                for name, val in wide_inputs.items():
                    val = tf.keras.layers.Flatten()(val)
                    if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                        raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                        .format(name, val.dtype))
                    elif val.dtype != self._dtype_policy.compute_dtype:
                        val = tf.cast(val, self._dtype_policy.compute_dtype,
                                        name=name + '_vec_cast' + self._dtype_policy.compute_dtype)
                    vec_inputs.append(val)
                    if name not in mmoe_inputs.keys(): # 对于wide侧和mmoe侧都有的特征就不会重复拼接了
                        cross_inputs.append(val)
                vec_inputs = tf.concat(vec_inputs, axis=-1, name='vec_features')
                cross_inputs = None if len(cross_inputs)==0 else tf.concat(cross_inputs, axis=-1, name='wide_features')
                wide_logits = None if self.wide_type!='LR' else self.wide_layer(vec_inputs)
        
        # feature cross before mmoe
        # 0: deep_feature->mmoe and mmoe_output->tower; 1: wide_feature+deep_feature->mmoe and mmoe_output->tower; 
        # 2: (same as 1) and wide_featrue+mmoe_output->tower; 3: (same as 1) and wide_logits+mmoe_output->tower
        # 4: (same as 0) and wide_feature+mmoe_output->tower; 5: (same as 0) and wide_logits+mmoe_output->tower
        if self.use_wide and self.feature_cross in [1,2,3]: # concat wide feature and deep feature
            # 如果使用wide侧特征, 且feature_cross为1,2,3
            # 则将wide侧特征(经过了input_layer或FMLayer)和mmoe侧特征(经过了input_layer)拼接, 并输入到mmoe_layer中(experts和gates)
            if self.wide_type!='FM' and cross_inputs is not None:
                concat_vals = tf.concat([concat_vals, cross_inputs], axis=-1, name='cross_features')
            if self.wide_type=='FM':
                concat_vals = tf.concat([concat_vals, wide_latent_matrix], axis=-1, name='cross_features')

        mmoe_outputs = self.mmoe_layer(concat_vals, training=training)
        task_outputs = []
        out_i = 0
        for mmoe_output, task_tower in zip(mmoe_outputs, self.task_towers):
            # feature cross before tower
            # 0: deep_feature->mmoe and mmoe_output->tower; 1: wide_feature+deep_feature->mmoe and mmoe_output->tower; 
            # 2: (same as 1) and wide_featrue+mmoe_output->tower; 3: (same as 1) and wide_logits+mmoe_output->tower
            # 4: (same as 0) and wide_feature+mmoe_output->tower; 5: (same as 0) and wide_logits+mmoe_output->tower
            # 如果使用wide侧特征, 且feature_cross为2或4, 则将wide侧特征(经过了input_layer或FMLayer)和mmoe_layer输出拼接, 并输入到task_layer中
            if self.use_wide and (self.feature_cross == 2 or self.feature_cross == 4):
                mmoe_output = tf.concat([mmoe_output, vec_inputs], axis=-1) if self.wide_type != 'FM' else tf.concat([mmoe_output, wide_latent_matrix], axis=-1)
            # 如果使用wide侧特征, 且feature_cross为3或5, 则将wide侧模型输出和mmoe_layer输出拼接, 并输入到task_layer中
            if self.use_wide and (self.feature_cross == 3 or self.feature_cross == 5):
                mmoe_output = tf.concat([mmoe_output, vec_inputs], axis=-1) if wide_logits is None else tf.concat([mmoe_output, wide_logits], axis=-1)
            
            task_output = task_tower(mmoe_output, training=training)
            if self.mixed_precision and task_output.dtype != tf.float32:
                task_output = tf.cast(task_output, tf.float32, self.name+"_mp_output_{}_cast2float32".format(out_i))
            task_outputs.append(task_output)
            out_i += 1

        if self.named_outputs:
            output_dict = {}
            for name, output, task_act_layer in zip(self.label_names, task_outputs, self.task_active_layers):
                # output cross
                # 如果设置output_cross为0, 则将mmoe的task_layer的输出和wide侧模型的输出进行加和, 再经过sigmoid得到最终输出
                if wide_logits is not None and self.output_cross==0: # 0: add; 1: concat;
                    output = output + wide_logits
                if task_act_layer:
                    output_dict[name] = task_act_layer(output)
                else:
                    output_dict[name] = tf.nn.sigmoid(output)
            return output_dict
        else:
            outputs = []
            for output, task_act_layer in zip(task_outputs, self.task_active_layers):
                 # 如果设置output_cross为0, 则将mmoe的task_layer的输出和wide侧模型的输出进行加和, 再经过sigmoid得到最终输出
                if wide_logits is not None and self.output_cross==0: # 0: add; 1: concat;
                    output = output + wide_logits
                if task_act_layer:
                    outputs.append(task_act_layer(output))
                else:
                    outputs.append(tf.nn.sigmoid(output))
            return tuple(outputs)

    def get_config(self):
        config = {
            'model_input_config': self.model_input_config,
            'task_structs': self.task_structs,
            'num_experts': self.num_experts,
            'expert_layers': self.expert_layers,
            'task_use_bias': self.task_use_bias,
            'task_hidden_act': self.task_hidden_act,
            'task_output_act': self.task_output_act,
            'task_use_bn': self.task_use_bn,
            'task_dropout': self.task_dropout,
            'task_l1_reg': self.task_l1_reg,
            'task_l2_reg': self.task_l2_reg,
            'expert_use_bias': self.expert_use_bias,
            'expert_act': self.expert_act,
            'expert_dropout': self.expert_dropout,
            'expert_use_bn': self.expert_use_bn,
            'expert_l1_reg': self.expert_l1_reg,
            'expert_l2_reg': self.expert_l2_reg,
            'gate_use_bias': self.gate_use_bias,
            'gate_l1_reg': self.gate_l1_reg,
            'gate_l2_reg': self.gate_l2_reg,
            'share_gates': self.share_gates,
            'wide_l1_reg': self.wide_l1_reg,
            'wide_l2_reg': self.wide_l2_reg,
            'use_wide': self.use_wide,
            'wide_type': self.wide_type,
            'wide_width': self.wide_width,
            'feature_cross': self.feature_cross,
            'output_cross': self.output_cross,
            'named_outputs': self.named_outputs,
            'name': self.name
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_save_signatures(self):
        if self.use_wide and self.wide_type=='FM':
            call_fn_specs_fm = self.wide_input_layer.get_tensor_specs()
            call_fn_specs = self.input_layer.get_tensor_specs()
            call_fn_specs = dict(call_fn_specs, **call_fn_specs_fm)
        else:
            call_fn_specs = self.input_layer.get_tensor_specs()
        
        sigs = {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.call.get_concrete_function(call_fn_specs)
        }
        return sigs
