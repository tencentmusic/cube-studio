

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('crowd-counting', 'damo/cv_hrnet_crowd-counting_dcanet')