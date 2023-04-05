

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('named-entity-recognition', 'damo/nlp_nested-ner_named-entity-recognition_chinese-base-med')