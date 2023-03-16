

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('live-category', 'damo/cv_resnet50_live-category')