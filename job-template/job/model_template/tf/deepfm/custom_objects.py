

from job.pkgs.tf.extend_layers import *
from job.pkgs.tf.feature_util import *
from job.model_template.tf.deepfm.model import DeepFMModel


custom_objects = {
    'tf': tf,
    'InputDesc': InputDesc,
    'FeatureProcessDesc': FeatureProcessDesc,
    'ModelInputConfig': ModelInputConfig,
    'ModelInputLayer': ModelInputLayer,
    'VocabLookupLayer': VocabLookupLayer,
    'PoolingEmbeddingLayer': PoolingEmbeddingLayer,
    'VocabMultiHotLayer': VocabMultiHotLayer,
    'VocabEmbeddingLayer': VocabEmbeddingLayer,
    'DNNLayer': DNNLayer,
    'FMLayer': FMLayer,
    'DeepFMModel': DeepFMModel
}
