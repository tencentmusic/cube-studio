

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('video-frame-interpolation', 'damo/cv_raft_video-frame-interpolation')