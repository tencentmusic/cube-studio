

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-object-segmentation', 'damo/cv_rdevos_video-object-segmentation')