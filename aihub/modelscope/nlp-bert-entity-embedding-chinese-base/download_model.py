

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-embedding', 'damo/nlp_bert_entity-embedding_chinese-base')