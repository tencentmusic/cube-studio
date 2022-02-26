
from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.mmoe.custom_objects import *


def awf_create_model_fn(model_input_config_file, task_structs, num_experts, expert_layers,
                        task_use_bias=True, task_hidden_act=None, task_output_act=None, task_use_bn=False,
                        task_dropout=None, task_l1_reg=None, task_l2_reg=None, expert_use_bias=True, expert_act='relu',
                        expert_dropout=None, expert_use_bn=False, expert_l1_reg=None, expert_l2_reg=None,
                        gate_use_bias=True, gate_l1_reg=None, gate_l2_reg=None, share_gates=False, named_outputs=None,
                        name='MMoE', pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = MMoEModel(model_input_config, task_structs, num_experts, expert_layers, task_use_bias,
                      task_hidden_act, task_output_act, task_use_bn, task_dropout, task_l1_reg, task_l2_reg,
                      expert_use_bias, expert_act, expert_dropout, expert_use_bn, expert_l1_reg, expert_l2_reg,
                      gate_use_bias, gate_l1_reg, gate_l2_reg, share_gates, named_outputs, name)
    return model


def awf_create_train_dataset_fn(model_input_config_file, data_file, batch_size, file_type=None, shuffle=False,
                                shuffle_buffer_size=None, drop_remainder=False, cache=False, repeat=False,
                                **kwargs):
    return create_dataset(model_input_config_file, data_file, file_type, None, batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, cache, repeat, **kwargs)


def awf_create_test_dataset_fn(model_input_config_file=None, data_file=None, batch_size=None, file_type=None,
                               shuffle=False, shuffle_buffer_size=None, drop_remainder=False, repeat=False,
                               **kwargs):
    if not model_input_config_file or not data_file:
        return None
    return create_dataset(model_input_config_file, data_file, file_type, None, batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, False, repeat, **kwargs)


def awf_create_val_dataset_fn(model_input_config_file=None, data_file=None, batch_size=None, file_type=None,
                              shuffle=False, shuffle_buffer_size=None, drop_remainder=False, repeat=False,
                              **kwargs):
    if not model_input_config_file or not data_file:
        return None
    return create_dataset(model_input_config_file, data_file, file_type, None, batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, False, repeat, **kwargs)


def awf_create_predict_dataset_fn(model_input_config_file=None, data_file=None, batch_size=None, file_type=None,
                                  shuffle=False, shuffle_buffer_size=None, drop_remainder=False, repeat=False,
                                  **kwargs):
    if not model_input_config_file or not data_file:
        return None
    return create_dataset(model_input_config_file, data_file, file_type, None, batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, False, repeat, **kwargs)


def awf_load_model_fn(path, name, model_input_config_file=None):
    model = tf.keras.models.load_model(path, custom_objects, compile=False)
    return model


def awf_model_to_save_fn(trained_model: MMoEModel):
    sigs = trained_model.get_save_signatures()
    return trained_model, sigs
