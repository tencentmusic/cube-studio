

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('image-depth-estimation', 'damo/cv_newcrfs_image-depth-estimation_indoor')