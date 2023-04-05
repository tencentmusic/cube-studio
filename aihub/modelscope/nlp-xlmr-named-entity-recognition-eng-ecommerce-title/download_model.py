

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('named-entity-recognition', 'damo/nlp_xlmr_named-entity-recognition_eng-ecommerce-title')