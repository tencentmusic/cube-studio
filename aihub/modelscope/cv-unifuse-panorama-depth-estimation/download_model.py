

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('panorama-depth-estimation', 'damo/cv_unifuse_panorama-depth-estimation')