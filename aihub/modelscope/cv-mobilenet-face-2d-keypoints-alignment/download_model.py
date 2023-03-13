

from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('damo/cv_mobilenet_face-2d-keypoints_alignment', cache_dir='/root/.cache/modelscope/hub/')
