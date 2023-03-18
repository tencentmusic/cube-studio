

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-deinterlace', 'damo/cv_unet_video-deinterlace')