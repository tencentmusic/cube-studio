

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('facial-expression-recognition', 'damo/cv_vgg19_facial-expression-recognition_fer')