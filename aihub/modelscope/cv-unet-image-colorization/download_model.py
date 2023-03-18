

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-colorization', 'damo/cv_unet_image-colorization')