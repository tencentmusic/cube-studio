

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-panoptic-segmentation', 'damo/cv_swinb_video-panoptic-segmentation_vipseg')