

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-similarity', 'damo/nlp_structbert_sentence-similarity_chinese-base')