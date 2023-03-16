

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('card-detection', 'damo/cv_resnet_carddetection_scrfd34gkps')