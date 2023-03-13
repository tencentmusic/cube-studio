

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-2d-keypoints', 'damo/cv_mobilenet_face-2d-keypoints_alignment')