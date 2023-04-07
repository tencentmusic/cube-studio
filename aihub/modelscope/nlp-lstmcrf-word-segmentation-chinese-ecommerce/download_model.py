

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('word-segmentation', 'damo/nlp_lstmcrf_word-segmentation_chinese-ecommerce')