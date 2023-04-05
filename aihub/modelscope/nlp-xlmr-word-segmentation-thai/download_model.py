

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('word-segmentation', 'damo/nlp_xlmr_word-segmentation_thai')