

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-attribute-recognition', 'damo/cv_resnet34_face-attribute-recognition_fairface')