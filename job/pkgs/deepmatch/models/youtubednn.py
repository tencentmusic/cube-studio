"""
Author:
    Weichen Shen, wcshen1994@163.com
Reference:
Covington P, Adams J, Sargin E. Deep neural networks for youtube recommendations[C]//Proceedings of the 10th ACM conference on recommender systems. 2016: 191-198.
"""
from deepctr.feature_column import build_input_features
from deepctr.layers import DNN
from deepctr.layers.utils import NoMask, combined_dnn_input
from tensorflow.python.keras.models import Model

from deepmatch.layers import PoolingLayer
from deepmatch.utils import get_item_embedding
from ..inputs import input_from_feature_columns, create_embedding_matrix
from ..layers.core import SampledSoftmaxLayer, EmbeddingIndex


def YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
               user_dnn_hidden_units=(64, 32),
               dnn_activation='relu', dnn_use_bn=False,
               l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', seed=1024, ):
    """Instantiates the YoutubeDNN Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param num_sampled: int, the number of classes to randomly sample per batch.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param output_activation: Activation function to use in output layer
    :return: A Keras model instance.

    """

    if len(item_feature_columns) > 1:
        raise ValueError("Now YoutubeNN only support 1 item feature like item_id")
    item_feature_name = item_feature_columns[0].name
    item_vocabulary_size = item_feature_columns[0].vocabulary_size

    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed)

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features, user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, output_activation=output_activation, seed=seed)(user_dnn_input)

    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_features[item_feature_name])

    item_embedding_matrix = embedding_matrix_dict[
        item_feature_name]
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))

    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])

    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [pooling_item_embedding_weight, user_dnn_out, item_features[item_feature_name]])
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)

    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("item_embedding",
                      get_item_embedding(pooling_item_embedding_weight, item_features[item_feature_name]))

    return model
