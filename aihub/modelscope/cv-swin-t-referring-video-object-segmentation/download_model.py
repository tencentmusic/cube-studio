

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('referring-video-object-segmentation', 'damo/cv_swin-t_referring_video-object-segmentation')