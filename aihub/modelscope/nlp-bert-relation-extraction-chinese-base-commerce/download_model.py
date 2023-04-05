

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('relation-extraction', 'damo/nlp_bert_relation-extraction_chinese-base-commerce')