

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('hand-2d-keypoints', 'damo/cv_hrnetw18_hand-pose-keypoints_coco-wholebody')