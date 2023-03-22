from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id="THUDM/chatglm-6b",token=os.getenv('HUGGINGFACE_TOKEN',None))