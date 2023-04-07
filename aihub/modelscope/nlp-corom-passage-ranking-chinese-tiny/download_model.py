

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-ranking', 'damo/nlp_corom_passage-ranking_chinese-tiny')