

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('named-entity-recognition', 'damo/nlp_raner_named-entity-recognition_chinese-large-generic')