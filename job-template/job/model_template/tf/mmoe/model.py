
from abc import ABC

from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_layers import ModelInputLayer, MMoELayer, DNNLayer
from job.pkgs.tf.helperfuncs import TF_REF_VERSION


class MMoEModel(tf.keras.Model, ABC):
    def __init__(self, model_input_config: ModelInputConfig, task_structs, num_experts, expert_layers,
                 task_use_bias=True, task_hidden_act=None, task_output_act=None, task_use_bn=False,
                 task_dropout=None, task_l1_reg=None, task_l2_reg=None, expert_use_bias=True, expert_act='relu',
                 expert_dropout=None, expert_use_bn=False, expert_l1_reg=None, expert_l2_reg=None, gate_use_bias=True,
                 gate_l1_reg=None, gate_l2_reg=None, share_gates=False, named_outputs=None, name='MMoE'):
        super(MMoEModel, self).__init__(name=name)

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
        idx = 0
        for struct, use_bias, hidden_act, output_act, use_bn, dropout, l1_reg, l2_reg \
                in zip(task_structs, task_use_bias_chk, task_hidden_act_chk, task_output_act_chk, task_use_bn_chk,
                       task_dropout_chk, task_l1_reg_chk, task_l2_reg_chk):
            self.task_towers.append(DNNLayer(struct, hidden_act, output_act, dropout, use_bn, l1_reg, l2_reg,
                                             use_bias, name='task_{}'.format(idx)))
            idx += 1

        self.mmoe_layer = MMoELayer(self.num_tasks, num_experts, expert_layers, expert_use_bias, expert_act,
                                    expert_dropout, expert_use_bn, expert_l1_reg, expert_l2_reg,
                                    gate_use_bias, gate_l1_reg, gate_l2_reg, share_gates)

        self.input_layer = ModelInputLayer(model_input_config, auto_create_embedding=True)

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
        self.share_gates = share_gates
        self.named_outputs = named_outputs
        if tf.__version__ < TF_REF_VERSION:
            self.mixed_precision = tf.keras.mixed_precision.experimental.get_layer_policy(self).loss_scale is not None
        else:
            self.mixed_precision = tf.keras.mixed_precision.global_policy().compute_dtype == tf.float16
        print("{}: mixed_precision={}".format(self.name, self.mixed_precision))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        transformed_inputs = self.input_layer(inputs)
        feat_vals = []
        for name, val in transformed_inputs.items():
            val = tf.keras.layers.Flatten()(val)
            if not val.dtype.is_integer and not val.dtype.is_floating and not val.dtype.is_bool:
                raise RuntimeError("dtype of input '{}' is {}, only float/int/bool are allowed"
                                   .format(name, val.dtype))
            elif val.dtype != self._dtype_policy.compute_dtype:
                val = tf.cast(val, self._dtype_policy.compute_dtype,
                              name=name + '_cast2' + self._dtype_policy.compute_dtype)
            feat_vals.append(val)
        concat_vals = tf.concat(feat_vals, axis=-1, name='concat_features')
        mmoe_outputs = self.mmoe_layer(concat_vals, training=training)
        task_outputs = []
        out_i = 0
        for mmoe_output, task_tower in zip(mmoe_outputs, self.task_towers):
            task_output = task_tower(mmoe_output, training=training)
            if self.mixed_precision and task_output.dtype != tf.float32:
                task_output = tf.cast(task_output, tf.float32, self.name+"_mp_output_{}_cast2float32".format(out_i))
            task_outputs.append(task_output)
            out_i += 1

        if self.named_outputs:
            output_dict = {}
            for name, output in zip(self.label_names, task_outputs):
                output_dict[name] = output
            return output_dict
        return tuple(task_outputs)

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
            'named_outputs': self.named_outputs,
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
