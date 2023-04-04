from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.feature_util import *
from job.model_template.tf.mmoe_v2.model import MMoEModelV2


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
    'FMLayer': FMLayer,
    'MMoELayer': MMoELayer,
    'MMoEModelV2': MMoEModelV2
}
