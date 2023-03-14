

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-human-matting', 'damo/cv_effnetv2_video-human-matting')