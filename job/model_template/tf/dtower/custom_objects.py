# coding=utf-8
# @Time     : 2021/1/25 18:50
# @Auther   : lionpeng@tencent.com

from job.model_template.tf.dtower.model import *
from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.extend_losses import *
from job.pkgs.tf.extend_metrics import *

custom_objects = {
    'tf': tf,
    'InputDesc': InputDesc,
    'FeatureProcessDesc': FeatureProcessDesc,
    'ModelInputConfig': ModelInputConfig,
    'VocabLookupLayer': VocabLookupLayer,
    'PoolingEmbeddingLayer': PoolingEmbeddingLayer,
    'VocabEmbeddingLayer': VocabEmbeddingLayer,
    'VocabMultiHotLayer': VocabMultiHotLayer,
    'ModelInputLayer': ModelInputLayer,
    'DNNLayer': DNNLayer,
    'BucktizeLayer': BucktizeLayer,
    'BPRLoss': BPRLoss,
    'PairHingeLoss': PairHingeLoss,
    'MeanMetricWrapper': MeanMetricWrapper,
    'PairOrderAccuracy': PairOrderAccuracy,
    'DTowerModel': DTowerModel
}
