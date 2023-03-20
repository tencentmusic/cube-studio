

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('ocr-recognition', 'damo/ofa_ocr-recognition_handwriting_base_zh')