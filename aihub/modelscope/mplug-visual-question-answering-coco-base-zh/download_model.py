

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('visual-question-answering', 'damo/mplug_visual-question-answering_coco_base_zh')