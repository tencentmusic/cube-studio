

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('human-wholebody-keypoint', 'damo/cv_hrnetw48_human-wholebody-keypoint_image')