

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-recognition', 'damo/cv_resnet_face-recognition_facemask')