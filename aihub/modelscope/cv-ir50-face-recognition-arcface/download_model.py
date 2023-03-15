

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-recognition', 'damo/cv_ir50_face-recognition_arcface')