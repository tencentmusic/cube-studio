'''
Author: your name
Date: 2021-06-15 15:36:30
LastEditTime: 2021-06-24 17:25:42
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /job-template/job/model_template/tf/ple/custom_objects.py
'''

from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.feature_util import *
from job.pkgs.tf.extend_metrics import GroupedAUC
from job.model_template.tf.ple.model import PLEModel, PLELayer
from job.model_template.tf.ple.custom.custom_layers import PersonalRadioInputDropoutV1, PersonalRadioExpertV1, PersonalRadioTowerV1

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
    'PersonalRadioInputDropoutV1': PersonalRadioInputDropoutV1,
    'PersonalRadioExpertV1': PersonalRadioExpertV1,
    'PersonalRadioTowerV1': PersonalRadioTowerV1,
    'DNNLayer': DNNLayer,
    'PLELayer': PLELayer,
    'PLEModel': PLEModel,
    'GroupedAUC': GroupedAUC
}
