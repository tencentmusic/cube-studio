

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-liveness', 'damo/cv_manual_face-liveness_flir')