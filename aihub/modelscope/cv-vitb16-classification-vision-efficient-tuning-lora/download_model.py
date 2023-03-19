

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('vision-efficient-tuning', 'damo/cv_vitb16_classification_vision-efficient-tuning-lora')