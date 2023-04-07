# coding=utf-8
# @Time     : 2021/1/12 21:24
# @Auther   : lionpeng@tencent.com

from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.youtube_dnn.custom_objects import *


def awf_create_model_fn(model_input_config_file, dnn_hidden_layers, dnn_hidden_act_fn='relu',
                        dnn_output_act_fn='relu', dnn_l1_reg=None, dnn_l2_reg=None, dnn_dropout=None,
                        dnn_use_bn=False, name='youtubednn', pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = YoutubeDNNModel(model_input_config, dnn_hidden_layers, dnn_hidden_act_fn, dnn_output_act_fn,
                            dnn_dropout, dnn_use_bn, dnn_l1_reg, dnn_l2_reg, name)
    return model


def awf_create_train_dataset_fn(model_input_config_file, data_file, batch_size, file_type=None, shuffle=False,
                                shuffle_buffer_size=None, drop_remainder=False, cache=False, repeat=False,
                                **kwargs):
    return create_dataset(model_input_config_file, data_file, file_type, ['user', 'item'], batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, cache, repeat, **kwargs)


def awf_create_test_dataset_fn(model_input_config_file=None, data_file=None, batch_size=None, file_type=None,
                               shuffle=False, shuffle_buffer_size=None, drop_remainder=False, repeat=False,
                               **kwargs):
    if not model_input_config_file or not data_file:
        return None
    return create_dataset(model_input_config_file, data_file, file_type, ['user', 'item'], batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, False, repeat, **kwargs)


def awf_create_val_dataset_fn(model_input_config_file=None, data_file=None, batch_size=None, file_type=None,
                              shuffle=False, shuffle_buffer_size=None, drop_remainder=False, repeat=False,
                              **kwargs):
    if not model_input_config_file or not data_file:
        return None
    return create_dataset(model_input_config_file, data_file, file_type, ['user', 'item'], batch_size, shuffle,
                          shuffle_buffer_size, drop_remainder, False, repeat, **kwargs)


def awf_model_to_save_fn(trained_model: YoutubeDNNModel):
    sigs = trained_model.get_save_signatures()
    return trained_model, sigs


def awf_load_model_fn(path, name, model_input_config_file=None):
    model = tf.keras.models.load_model(path, custom_objects, compile=False)
    return model


def sampled_softmax_loss(model: YoutubeDNNModel, num_samples, sample_algo=None, sample_algo_params=None):
    num_classes = model.get_item_cardinality()
    loss = SampledSoftmaxLoss(model, num_samples, num_classes+1, sample_algo, sample_algo_params)
    return loss


def topk_hitrate(model: YoutubeDNNModel, k=10, name="topk_hitrate"):
    num_classes = model.get_item_cardinality()
    metric = TopKHitrate(model, num_classes, k, name)
    return metric
