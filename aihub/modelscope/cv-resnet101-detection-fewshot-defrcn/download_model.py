

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-fewshot-detection', 'damo/cv_resnet101_detection_fewshot-defrcn')