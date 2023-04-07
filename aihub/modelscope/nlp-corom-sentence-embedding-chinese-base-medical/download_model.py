

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-embedding', 'damo/nlp_corom_sentence-embedding_chinese-base-medical')