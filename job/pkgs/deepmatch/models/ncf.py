"""
Author:
    Jieyu Yang , yangjieyu@zju.edu.cn

Reference:
He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.
"""

import math

from deepctr.feature_column import input_from_feature_columns, build_input_features, SparseFeat
from deepctr.layers import DNN, combined_dnn_input
from tensorflow.python.keras.layers import Lambda, Concatenate, Multiply
from tensorflow.python.keras.models import Model


def NCF(user_feature_columns, item_feature_columns, user_gmf_embedding_dim=20, item_gmf_embedding_dim=20,
        user_mlp_embedding_dim=20, item_mlp_embedding_dim=20, dnn_use_bn=False,
        dnn_hidden_units=(64, 32), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0,
        seed=1024):
    """Instantiates the NCF Model architecture.

    :param user_feature_columns: A dict containing user's features and features'dim.
    :param item_feature_columns: A dict containing item's features and features'dim.
    :param user_gmf_embedding_dim: int.
    :param item_gmf_embedding_dim: int.
    :param user_mlp_embedding_dim: int.
    :param item_mlp_embedding_dim: int.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :return: A Keras model instance.

    """

    user_dim = len(user_feature_columns) * user_gmf_embedding_dim
    item_dim = len(item_feature_columns) * item_gmf_embedding_dim
    dim = (user_dim * item_dim) / (math.gcd(user_dim, item_dim))
    user_gmf_embedding_dim = int(dim / len(user_feature_columns))
    item_gmf_embedding_dim = int(dim / len(item_feature_columns))

    # Generalized Matrix Factorization (GMF) Part
    user_gmf_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=user_gmf_embedding_dim)
                                for feat, size in user_feature_columns.items()]
    user_features = build_input_features(user_gmf_feature_columns)
    user_inputs_list = list(user_features.values())
    user_gmf_sparse_embedding_list, user_gmf_dense_value_list = input_from_feature_columns(user_features,
                                                                                           user_gmf_feature_columns,
                                                                                           l2_reg_embedding, seed=seed,
                                                                                           prefix='gmf_')
    user_gmf_input = combined_dnn_input(user_gmf_sparse_embedding_list, [])
    user_gmf_out = Lambda(lambda x: x, name="user_gmf_embedding")(user_gmf_input)

    item_gmf_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=item_gmf_embedding_dim)
                                for feat, size in item_feature_columns.items()]
    item_features = build_input_features(item_gmf_feature_columns)
    item_inputs_list = list(item_features.values())
    item_gmf_sparse_embedding_list, item_gmf_dense_value_list = input_from_feature_columns(item_features,
                                                                                           item_gmf_feature_columns,
                                                                                           l2_reg_embedding, seed=seed,
                                                                                           prefix='gmf_')
    item_gmf_input = combined_dnn_input(item_gmf_sparse_embedding_list, [])
    item_gmf_out = Lambda(lambda x: x, name="item_gmf_embedding")(item_gmf_input)

    gmf_out = Multiply()([user_gmf_out, item_gmf_out])

    # Multi-Layer Perceptron (MLP) Part
    user_mlp_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=user_mlp_embedding_dim)
                                for feat, size in user_feature_columns.items()]
    user_mlp_sparse_embedding_list, user_mlp_dense_value_list = input_from_feature_columns(user_features,
                                                                                           user_mlp_feature_columns,
                                                                                           l2_reg_embedding, seed=seed,
                                                                                           prefix='mlp_')
    user_mlp_input = combined_dnn_input(
        user_mlp_sparse_embedding_list, user_mlp_dense_value_list)
    user_mlp_out = Lambda(lambda x: x, name="user_mlp_embedding")(user_mlp_input)

    item_mlp_feature_columns = [SparseFeat(feat, vocabulary_size=size, embedding_dim=item_mlp_embedding_dim)
                                for feat, size in item_feature_columns.items()]

    item_mlp_sparse_embedding_list, item_mlp_dense_value_list = input_from_feature_columns(item_features,
                                                                                           item_mlp_feature_columns,
                                                                                           l2_reg_embedding, seed=seed,
                                                                                           prefix='mlp_')
    item_mlp_input = combined_dnn_input(
        item_mlp_sparse_embedding_list, item_mlp_dense_value_list)
    item_mlp_out = Lambda(lambda x: x, name="item_mlp_embedding")(item_mlp_input)

    mlp_input = Concatenate(axis=1)([user_mlp_out, item_mlp_out])
    mlp_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  dnn_use_bn, seed = seed, name="mlp_embedding")(mlp_input)

    # Fusion of GMF and MLP
    neumf_input = Concatenate(axis=1)([gmf_out, mlp_out])
    neumf_out = DNN(hidden_units=[1], activation='sigmoid',seed=seed)(neumf_input)
    output = Lambda(lambda x: x, name='neumf_out')(neumf_out)

    # output = PredictionLayer(task, False)(neumf_out)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    return model
