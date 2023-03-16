

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-category', 'damo/cv_resnet50_video-category')