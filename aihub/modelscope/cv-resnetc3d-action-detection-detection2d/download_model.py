

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('action-detection', 'damo/cv_ResNetC3D_action-detection_detection2d')