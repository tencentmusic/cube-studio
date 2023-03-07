
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

text_generation_zh = pipeline(Tasks.text_generation, model='damo/nlp_gpt3_text-generation_1.3B')

