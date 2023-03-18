

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-colorization', 'damo/cv_unet_video-colorization')