

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-generation', 'damo/nlp_gpt3_kuakua-robot_chinese-large')