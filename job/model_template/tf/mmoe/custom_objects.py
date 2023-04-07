# coding=utf-8
# @Time     : 2021/1/25 18:50
# @Auther   : lionpeng@tencent.com

from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.feature_util import *
from job.model_template.tf.mmoe.model import MMoEModel


custom_objects = {
    'tf': tf,
    'InputDesc': InputDesc,
    'FeatureProcessDesc': FeatureProcessDesc,
    'ModelInputConfig': ModelInputConfig,
    'ModelInputLayer': ModelInputLayer,
    'VocabLookupLayer': VocabLookupLayer,
    'PoolingEmbeddingLayer': PoolingEmbeddingLayer,
    'VocabEmbeddingLayer': VocabEmbeddingLayer,
    'VocabMultiHotLayer': VocabMultiHotLayer,
    'DNNLayer': DNNLayer,
    'MMoELayer': MMoELayer,
    'MMoEModel': MMoEModel
}
