

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('nli', 'damo/nlp_structbert_fact-checking_chinese-base')