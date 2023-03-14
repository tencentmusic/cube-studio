

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('ocr-recognition', 'damo/cv_crnn_ocr-recognition-general_damo')