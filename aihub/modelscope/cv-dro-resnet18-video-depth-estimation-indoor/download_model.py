

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-depth-estimation', 'damo/cv_dro-resnet18_video-depth-estimation_indoor')