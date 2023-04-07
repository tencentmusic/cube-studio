'''
Author: your name
Date: 2021-08-13 09:24:03
LastEditTime: 2021-08-25 17:32:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/din/template.py
'''
from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.din.custom_objects import *


def awf_create_model_fn(model_input_config_file, 
                        attn_units=[36,1], attn_act=['prelu'], attn_mode='SUM', attn_norm=True,
                        dnn_units=[200,80,2], dnn_act=['prelu','prelu'], 
                        dnn_dropout=None, dnn_use_bn=None, dnn_l1_reg=None, dnn_l2_reg=None, 
                        name='DIN', pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = DINModel(model_input_config, attn_units, attn_act, attn_mode, attn_norm, 
                     dnn_units, dnn_act, dnn_dropout, dnn_use_bn, dnn_l1_reg, dnn_l2_reg, name)
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


def awf_model_to_save_fn(trained_model: DINModel):
    sigs = trained_model.get_save_signatures()
    return trained_model, sigs
