

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-object-detection', 'damo/cv_cspnet_video-object-detection_streamyolo')