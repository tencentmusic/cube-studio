

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-inpainting', 'damo/cv_fft_inpainting_lama')