

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('task-oriented-conversation', 'damo/nlp_space_dialog-state-tracking')