

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('text-to-image-synthesis', 'damo/multi-modal_chinese_stable_diffusion_v1.0')