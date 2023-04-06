'''
Author: your name
Date: 2021-06-09 17:07:40
LastEditTime: 2021-08-25 17:33:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/ple/template.py
'''
from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.ple.custom_objects import *


def awf_create_model_fn(model_input_config_file, ordered_task_names, layer_number, ple_dict, tower_dict,
                        tower_dependencies_dict={},
                        dropout_layer="PersonalRadioInputDropoutV1",
                        custom_layer_file_path="/app/job/model_template/tf/ple/custom/custom_layers.py",
                        is_concat_gate_input=True, use_inputs_dropout=True, name='ple',
                        pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = PLEModel(model_input_config, ordered_task_names, layer_number, ple_dict, tower_dict, 
                        tower_dependencies_dict, dropout_layer,
                        custom_layer_file_path, is_concat_gate_input, use_inputs_dropout, name)
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


def awf_model_to_save_fn(trained_model: PLEModel):
    sigs = trained_model.get_save_signatures()
    return trained_model, sigs
