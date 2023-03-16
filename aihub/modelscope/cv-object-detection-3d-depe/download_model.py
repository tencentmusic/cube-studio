

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('object-detection-3d', 'damo/cv_object-detection-3d_depe')