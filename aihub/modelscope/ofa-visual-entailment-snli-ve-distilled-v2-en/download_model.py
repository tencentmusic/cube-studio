

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('visual-entailment', 'damo/ofa_visual-entailment_snli-ve_distilled_v2_en')