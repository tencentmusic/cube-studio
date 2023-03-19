

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('domain-specific-object-detection', 'damo/cv_yolox-pai_hand-detection')