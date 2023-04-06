'''
Author: your name
Date: 2021-08-13 09:24:03
LastEditTime: 2021-08-25 17:31:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/comirec/template.py
'''
from job.model_template.tf.data_helper import create_dataset
from job.model_template.tf.comirec.custom_objects import *


def awf_create_model_fn(model_input_config_file, 
                        item_embedding_dim, seq_max_len, interest_extractor='DR', 
                        num_interests=3, hidden_size=None, capsule_size=None, add_pos=None, pow_p=1,
                        name='ComiRec', pack_path=None, data_path=None):
    model_input_config = ModelInputConfig.parse(model_input_config_file, pack_path, data_path)
    model = ComiRecModel(model_input_config, item_embedding_dim, seq_max_len, interest_extractor, 
                         num_interests, hidden_size, capsule_size, add_pos, pow_p, name)
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


def awf_load_model_fn(path, name, model_input_config_file=None, purpose=None):
    model = tf.keras.models.load_model(path, custom_objects, compile=False)
    # 由于模型文件夹下会保存MIND本体和Item Embedding导出模型, 在验证或预测时如果需要用MIND模型本体来验证模型效果, 则不加载模型名称结尾是item_embedding_model的模型
    if purpose == 'evaluate' and (model.name.endswith('-item_embeddings_model') or model.name.endswith('-user_model')):
        del model
        return None
    if purpose == 'predict' and not model.name.endswith('-item_embeddings_model'):
        del model
        return None
    return model


def awf_model_to_save_fn(trained_model: ComiRecModel):
    # sigs = trained_model.get_save_signatures()
    user_interests_model = trained_model.get_user_interests_model()
    item_embeddings_model = trained_model.get_item_embeddings_model()

    # return [(trained_model, sigs), item_embeddings_model]
    return [trained_model, user_interests_model, item_embeddings_model]

def sampled_softmax_loss(model: ComiRecModel, num_samples, sample_algo=None, sample_algo_params=None):
    label = model.get_item_label()
    num_classes = model.get_item_feature_cardinality(label.embedding_name) # label name must be the same with embedding name of feature hist sequence
    loss = SampledSoftmaxLoss(model, num_samples, num_classes+1, sample_algo, sample_algo_params)
    return loss


def topk_hitrate(model: ComiRecModel, k=10, name="topk_hitrate"):
    label = model.get_item_label()
    num_classes = model.get_item_feature_cardinality(label.embedding_name)
    metric = TopKHitrate(model, num_classes, k, name)
    return metric