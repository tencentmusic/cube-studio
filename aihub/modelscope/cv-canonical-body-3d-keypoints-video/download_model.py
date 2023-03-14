

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('body-3d-keypoints', 'damo/cv_canonical_body-3d-keypoints_video')