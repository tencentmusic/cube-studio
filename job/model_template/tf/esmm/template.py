'''
Author: your name
Date: 2021-08-13 09:24:03
LastEditTime: 2021-08-25 17:32:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/esmm/template.py
'''
# coding=utf-8
# @Time     : 2021/1/12 21:24
# @Auther   : lionpeng@tencent.com

from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.esmm.custom_objects import *


def awf_create_model_fn(model_input_config_file, cvr_dnn_units=[1], cvr_dnn_hidden_act=None,
                        cvr_dnn_use_bias=True, cvr_dnn_use_bn=False, cvr_dnn_dropout=None,
                        cvr_dnn_l1_reg=None, cvr_dnn_l2_reg=None, ctr_dnn_units=[1], ctr_dnn_hidden_act=None,
                        ctr_dnn_use_bias=True, ctr_dnn_use_bn=False, ctr_dnn_dropout=None,
                        ctr_dnn_l1_reg=None, ctr_dnn_l2_reg=None, ctr_label_name='', ctcvr_label_name='',
                        name="esmm", pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = ESMMModel(model_input_config, cvr_dnn_units, cvr_dnn_hidden_act, cvr_dnn_use_bias,
                      cvr_dnn_use_bn, cvr_dnn_dropout, cvr_dnn_l1_reg, cvr_dnn_l2_reg, ctr_dnn_units,
                      ctr_dnn_hidden_act, ctr_dnn_use_bias, ctr_dnn_use_bn, ctr_dnn_dropout, ctr_dnn_l1_reg,
                      ctr_dnn_l2_reg, ctr_label_name, ctcvr_label_name, name)
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


def awf_model_to_save_fn(trained_model: ESMMModel):
    sigs = trained_model.get_call_signatures()
    cvr_model = trained_model.get_cvr_model()
    ctr_model = trained_model.get_ctr_model()
    return [(trained_model, sigs), cvr_model, ctr_model]


def awf_load_model_fn(path, name, model_input_config_file=None, purpose=None):
    model = tf.keras.models.load_model(path, custom_objects, compile=False)
    if purpose == 'evaluate' and (model.name.endswith('-cvr') or model.name.endswith('-ctr')):
        del model
        return None
    return model
