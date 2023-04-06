

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('fill-mask', 'damo/nlp_medical_structbert_backbone_base')