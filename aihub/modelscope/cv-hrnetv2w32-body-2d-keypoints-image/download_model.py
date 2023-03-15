

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('body-2d-keypoints', 'damo/cv_hrnetv2w32_body-2d-keypoints_image')