

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-detection', 'damo/cv_resnet50_face-detection_retinaface')