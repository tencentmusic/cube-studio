

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('sentence-similarity', 'damo/nlp_masts_sentence-similarity_clue_chinese-large')