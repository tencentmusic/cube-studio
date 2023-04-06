

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-ranking', 'damo/nlp_bert_entity-matching_chinese-base')