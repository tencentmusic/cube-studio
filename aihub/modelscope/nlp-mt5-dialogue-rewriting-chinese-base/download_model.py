

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text2text-generation', 'damo/nlp_mt5_dialogue-rewriting_chinese-base')