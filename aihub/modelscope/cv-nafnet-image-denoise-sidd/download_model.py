

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-denoising', 'damo/cv_nafnet_image-denoise_sidd')