

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-summarization', 'damo/cv_googlenet_pgl-video-summarization')