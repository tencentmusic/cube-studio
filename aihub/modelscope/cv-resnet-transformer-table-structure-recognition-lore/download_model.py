

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('lineless-table-recognition', 'damo/cv_resnet-transformer_table-structure-recognition_lore')