

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('zero-shot-classification', 'damo/nlp_structbert_zero-shot-classification_chinese-large')