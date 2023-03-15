

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-human-hand-detection', 'damo/cv_nanodet_face-human-hand-detection')