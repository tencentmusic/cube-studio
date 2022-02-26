
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
