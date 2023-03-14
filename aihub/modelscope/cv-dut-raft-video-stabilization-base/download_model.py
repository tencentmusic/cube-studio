

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-stabilization', 'damo/cv_dut-raft_video-stabilization_base')