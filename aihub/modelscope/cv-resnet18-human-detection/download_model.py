

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('human-detection', 'damo/cv_resnet18_human-detection')