

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('license-plate-detection', 'damo/cv_resnet18_license-plate-detection_damo')