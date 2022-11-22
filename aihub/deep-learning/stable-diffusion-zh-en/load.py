import io, sys, os
import random
from datetime import datetime
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type
import torch
import pysnooper
from flagai.auto_model.auto_loader import AutoLoader

os.makedirs('/modeldir', exist_ok=True)
loader = AutoLoader(task_name="text2img",  # contrastive learning
                    model_name="AltDiffusion-m9",
                    model_dir="/modeldir")
model = loader.get_model()
