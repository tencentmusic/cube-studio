

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-quality-assessment-mos', 'damo/cv_resnet_image-quality-assessment-mos_youtubeUGC')