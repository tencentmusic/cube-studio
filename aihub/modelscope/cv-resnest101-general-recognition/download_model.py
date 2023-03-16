

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('general-recognition', 'damo/cv_resnest101_general_recognition')