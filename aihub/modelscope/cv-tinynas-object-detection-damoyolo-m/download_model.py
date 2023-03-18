

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-object-detection', 'damo/cv_tinynas_object-detection_damoyolo-m')