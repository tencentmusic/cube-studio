

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-image-synthesis', 'damo/cv_cartoon_stable_diffusion_illustration')