

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('animal-recognition', 'damo/cv_resnest101_animal_recognition')