

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('ocr-recognition', 'damo/cv_convnextTiny_ocr-recognition-general_damo')