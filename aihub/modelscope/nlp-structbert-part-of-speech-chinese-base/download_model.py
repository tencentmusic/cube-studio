

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('part-of-speech', 'damo/nlp_structbert_part-of-speech_chinese-base')