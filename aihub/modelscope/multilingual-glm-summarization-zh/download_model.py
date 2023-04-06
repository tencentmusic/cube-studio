

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-summarization', 'ZhipuAI/Multilingual-GLM-Summarization-zh')