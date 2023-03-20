

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('token-classification', 'damo/mgeo_geographic_elements_tagging_chinese_base')