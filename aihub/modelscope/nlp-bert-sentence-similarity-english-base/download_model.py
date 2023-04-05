

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-similarity', 'damo/nlp_bert_sentence-similarity_english-base')