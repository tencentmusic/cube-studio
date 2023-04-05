from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.feature_util import *
from job.model_template.tf.din.model import DINModel


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
    'AttentionLayer': AttentionLayer,
    'LocalActivationUnit': LocalActivationUnit,
    'DINModel': DINModel
}
