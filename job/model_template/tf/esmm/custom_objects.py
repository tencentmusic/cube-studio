# coding=utf-8
# @Time     : 2021/1/25 18:50
# @Auther   : lionpeng@tencent.com

from job.model_template.tf.esmm.model import *
from job.pkgs.tf.extend_layers import *

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
    'ESMMModel': ESMMModel
}
