

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('language-guided-video-summarization', 'damo/cv_clip-it_video-summarization_language-guided_en')