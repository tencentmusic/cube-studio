

from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('{{APP_PATH}}', cache_dir='/root/.cache/modelscope/hub/')

