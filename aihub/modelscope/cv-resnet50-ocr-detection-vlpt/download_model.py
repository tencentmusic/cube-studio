

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('ocr-detection', 'damo/cv_resnet50_ocr-detection-vlpt')