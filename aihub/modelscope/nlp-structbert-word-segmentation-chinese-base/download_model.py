

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('word-segmentation', 'damo/nlp_structbert_word-segmentation_chinese-base')