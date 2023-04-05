'''
Author: your name
Date: 2021-08-13 09:24:03
LastEditTime: 2021-08-25 17:32:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/dtower/template.py
'''
# coding=utf-8
# @Time     : 2021/1/12 21:24
# @Auther   : lionpeng@tencent.com

from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.dtower.custom_objects import *


def awf_create_model_fn(model_input_config_file, user_tower_units=[], user_tower_hidden_act='relu',
                        user_tower_output_act=None, user_tower_dropout=None, user_tower_use_bn=None,
                        user_tower_use_ln=None, user_tower_l1_reg=None, user_tower_l2_reg=None, 
                        user_tower_use_bias=False, item_tower_units=[], item_tower_hidden_act='relu', 
                        item_tower_output_act=None, item_tower_dropout=None, item_tower_use_bn=None, 
                        item_tower_use_ln=None, item_tower_l1_reg=None, item_tower_l2_reg=None, 
                        item_tower_use_bias=False, pairwise=False, use_cosine=False,
                        name='dtower', pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = DTowerModel(model_input_config, user_tower_units, user_tower_hidden_act, user_tower_output_act,
                        user_tower_dropout, user_tower_use_bn, user_tower_use_ln, user_tower_l1_reg, user_tower_l2_reg,
                        user_tower_use_bias, item_tower_units, item_tower_hidden_act, item_tower_output_act,
                        item_tower_dropout, item_tower_use_bn, item_tower_use_ln, item_tower_l1_reg, item_tower_l2_reg,
                        item_tower_use_bias, pairwise, use_cosine, name)
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


def awf_model_to_save_fn(trained_model: DTowerModel):
    sigs = trained_model.get_call_signatures()
    user_model = trained_model.get_user_model()
    item_model = trained_model.get_item_model()
    ctr_model = trained_model.get_ctr_model()
    return [(trained_model, sigs), user_model, item_model, ctr_model]


def awf_load_model_fn(path, name, model_input_config_file=None, purpose=None):
    model = tf.keras.models.load_model(path, custom_objects, compile=False)
    if purpose == 'evaluate' and (model.name.endswith('-user_tower') or model.name.endswith('-item_tower') or
                                  model.name.endswith('-ctr_model')):
        del model
        return None
    return model
