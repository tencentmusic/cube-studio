

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('table-recognition', 'damo/cv_dla34_table-structure-recognition_cycle-centernet')