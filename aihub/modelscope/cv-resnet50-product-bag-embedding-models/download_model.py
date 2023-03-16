

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('product-retrieval-embedding', 'damo/cv_resnet50_product-bag-embedding-models')