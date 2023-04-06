

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('extractive-summarization', 'damo/nlp_ponet_extractive-summarization_topic-level_chinese-base')