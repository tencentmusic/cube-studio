

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('fill-mask', 'damo/nlp_veco_fill-mask-large')