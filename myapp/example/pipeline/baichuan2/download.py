# pip install modelscope

from modelscope.hub.snapshot_download import snapshot_download
import os
local_dir_root = os.path.abspath(__file__)
local_dir_root = os.path.dirname(local_dir_root)
snapshot_download('baichuan-inc/Baichuan2-13B-Chat', cache_dir=local_dir_root)