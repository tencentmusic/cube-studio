

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text2text-generation', 'ClueAI/ChatYuan-large')