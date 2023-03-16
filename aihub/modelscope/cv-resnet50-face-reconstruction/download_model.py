

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-reconstruction', 'damo/cv_resnet50_face-reconstruction')