import os
import torch
import logging
import os
import platform
import socket
import sys
import time
from dotmap import DotMap
from hydra_node.models import StableDiffusionModel, DalleMiniModel, BasedformerModel, EmbedderModel
from hydra_node import lowvram
import traceback
import zlib
from pathlib import Path
from ldm.modules.attention import CrossAttention, HyperLogic

model_map = {
    "stable-diffusion": StableDiffusionModel,
    "dalle-mini": DalleMiniModel,
    "basedformer": BasedformerModel,
    "embedder": EmbedderModel,
    }

def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result

def crc32(filename, chunksize=65536):
    """Compute the CRC-32 checksum of the contents of the given filename"""
    with open(filename, "rb") as f:
        checksum = 0
        while True:
            chunk = f.read(chunksize)
            if not chunk:
              break
            checksum = zlib.crc32(chunk, checksum)
        #while (chunk := f.read(chunksize)) :
        #    checksum = zlib.crc32(chunk, checksum)
        return '%08X' % (checksum & 0xFFFFFFFF)

def load_modules(path):
    path = Path(path)
    modules = {}
    if not path.is_dir():
        return

    for file in path.iterdir():
        module = load_module(file, "cpu")
        modules[file.stem] = module
        print(f"Loaded module {file.stem}")

    return modules

def load_module(path, device):
    path = Path(path)
    if not path.is_file():
        print("Module path {} is not a file".format(path))

    network = {
        768: (HyperLogic(768).to(device), HyperLogic(768).to(device)),
        1280: (HyperLogic(1280).to(device), HyperLogic(1280).to(device)),
        640: (HyperLogic(640).to(device), HyperLogic(640).to(device)),
        320: (HyperLogic(320).to(device), HyperLogic(320).to(device)),
    }

    state_dict = torch.load(path)
    for key in state_dict.keys():
        network[key][0].load_state_dict(state_dict[key][0])
        network[key][1].load_state_dict(state_dict[key][1])

    return network

def init_config_model():
    config = DotMap()
    config.savefiles = os.getenv("SAVE_FILES", False)
    config.dtype = os.getenv("DTYPE", "float32")
    config.device = os.getenv("DEVICE", "cuda")
    config.amp = os.getenv("AMP", "1")
    if config.amp == "1":
        config.amp = True
    elif config.amp == "0":
        config.amp = False

    is_dev = ""
    environment = "production"
    config.is_dev = is_dev

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    config.logger = logger

    # Gather node information
    config.cuda_dev = torch.cuda.current_device()
    cpu_id = platform.processor()
    if os.path.exists('/proc/cpuinfo'):
        cpu_id = [line for line in open("/proc/cpuinfo", 'r').readlines() if
         'model name' in line][0].rstrip().split(': ')[-1]

    config.cpu_id = cpu_id
    config.gpu_id = torch.cuda.get_device_name(config.cuda_dev)
    config.node_id = platform.node()

    # Report on our CUDA memory and model.
    gb_gpu = int(torch.cuda.get_device_properties(
        config.cuda_dev).total_memory / (1000 * 1000 * 1000))
    logger.info(f"CPU: {config.cpu_id}")
    logger.info(f"GPU: {config.gpu_id}")
    logger.info(f"GPU RAM: {gb_gpu}gb")

    config.model_name = "stable-diffusion"
    logger.info(f"MODEL: {config.model_name}")

    # Resolve where we get our model and data from.
    config.model_path = os.getenv('MODEL_PATH', "models/animefull-final-pruned")
    config.enable_ema = os.getenv('ENABLE_EMA', "1")
    config.basedformer = os.getenv('BASEDFORMER', "0")
    config.penultimate = os.getenv('PENULTIMATE', "0")
    config.vae_path = os.getenv('VAE_PATH', "models/animevae.pt")
    config.module_path = os.getenv('MODULE_PATH', "models/modules")
    config.prior_path = os.getenv('PRIOR_PATH', "models/vector_adjust_v2.pt")
    config.default_config = os.getenv('DEFAULT_CONFIG', None)
    config.quality_hack = os.getenv('QUALITY_HACK', "0")
    config.clip_contexts = os.getenv('CLIP_CONTEXTS', 3)
    try:
        config.clip_contexts = int(config.clip_contexts)
        if config.clip_contexts < 1 or config.clip_contexts > 10:
            config.clip_contexts = 1
    except:
        config.clip_contexts = 1

    # Misc settings
    config.model_alias = os.getenv('MODEL_ALIAS')

    # Instantiate our actual model.
    load_time = time.time()
    model_hash = None

    try:
        if config.model_name != "dalle-mini":
            model = no_init(lambda: model_map[config.model_name](config))
        else:
            model = model_map[config.model_name](config)

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to load model: {str(e)}")
        #exit gunicorn
        sys.exit(4)

    if config.model_name == "stable-diffusion":
        folder = Path(config.model_path)
        if (folder / "pruned.ckpt").is_file():
            model_path = folder / "pruned.ckpt"
        else:
            model_path = folder / "model.ckpt"
        model_hash = crc32(model_path)

        #Load Modules
        if config.module_path is not None:
            modules = load_modules(config.module_path)
            #attach it to the model
            model.premodules = modules

    lowvram.setup_for_low_vram(model.model, True)

    config.model = model

    time_load = time.time() - load_time
    logger.info(f"Models loaded in {time_load:.2f}s")

    return model, config, model_hash
