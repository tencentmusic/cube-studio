

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('zero-shot-classification', 'damo/nlp_bert_zero-shot_english-base')