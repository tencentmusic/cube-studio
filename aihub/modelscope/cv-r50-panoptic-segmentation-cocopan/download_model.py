

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-segmentation', 'damo/cv_r50_panoptic-segmentation_cocopan')