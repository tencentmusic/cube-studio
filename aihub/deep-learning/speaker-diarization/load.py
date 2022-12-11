from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id="pyannote/speaker-diarization",revision='2.1',token=os.getenv('HUGGINGFACE_TOKEN',None))
