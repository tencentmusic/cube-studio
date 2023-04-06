

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('protein-structure', 'DPTech/uni-fold-monomer')