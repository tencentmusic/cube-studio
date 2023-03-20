

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('ocr-recognition', 'damo/ofa_ocr-recognition_document_base_zh')