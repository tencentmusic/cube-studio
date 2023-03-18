

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-detection', 'damo/cv_resnet101_face-detection_cvpr22papermogface')