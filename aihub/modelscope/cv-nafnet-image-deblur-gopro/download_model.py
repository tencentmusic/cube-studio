

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-deblurring', 'damo/cv_nafnet_image-deblur_gopro')