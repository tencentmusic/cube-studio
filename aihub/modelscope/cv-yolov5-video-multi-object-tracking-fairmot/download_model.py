

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-multi-object-tracking', 'damo/cv_yolov5_video-multi-object-tracking_fairmot')