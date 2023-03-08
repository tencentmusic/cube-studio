

from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('damo/nlp_gpt3_text-generation_1.3B', cache_dir='/root/.cache/modelscope/hub/')

