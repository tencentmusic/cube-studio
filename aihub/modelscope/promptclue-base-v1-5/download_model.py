

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text2text-generation', 'ClueAI/PromptCLUE-base-v1-5')