

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('face-image-generation', 'damo/cv_gan_face-image-generation')