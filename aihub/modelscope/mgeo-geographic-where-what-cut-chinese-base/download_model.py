

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('token-classification', 'damo/mgeo_geographic_where_what_cut_chinese_base')