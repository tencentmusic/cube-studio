

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('movie-scene-segmentation', 'damo/cv_resnet50-bert_video-scene-segmentation_movienet')