import base64
import io, sys, os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type

import pysnooper
import os
import argparse, os, re
import datetime

import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from Utils import split_weighted_subprompts


class Txt2Img_Model(Model):
    # 模型基础信息定义
    name = 'txt2img'
    label = '文字转图像'
    description = "ai示例应用，详细描述，都会显示应用描述上，支持markdown"
    field = "神经网络"
    scenes = "图像创作"
    status = 'online'
    version = 'v20221022'
    doc = 'https://github.com/tencentmusic/cube-studio/tree/master/aihub'  # 'https://帮助文档的链接地址'
    pic = 'https://user-images.githubusercontent.com/20157705/170216784-91ac86f7-d272-4940-a285-0c27d6f6cd96.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    # 运行基础环境脚本
    init_shell = 'init.sh'

    inference_inputs = [
        Field(type=Field_type.text, name='prompt', label='输入的文字内容',
              describe='输入的文字内容，暂仅支持英文输入~', default='a photograph of an astronaut riding a horse'),
        Field(type=Field_type.int, name='ddim_steps', label='推理的次数',
              describe='推理进行的次数，推荐20-50次将会得到更接近真实的图片', default=50),
        Field(type=Field_type.int, name='n_samples', label='推理出的图像数量(不支持修改!)',
              describe='结果中所展示的图片数量，数量越多则会导致性能下降', default=1),
        Field(type=Field_type.int, name='seed', label='初始化的种子',
              describe='不同的种子会得到不同的结果，理解为一种随机数吧~', default=None)
    ]

    # 加载模型
    def load_model(self):
        # self.model = load("/xxx/xx/a.pth")
        pl_sd = torch.load('full-model.ckpt', map_location="cpu")
        sd = pl_sd["state_dict"]
        return sd

    # 推理
    @pysnooper.snoop()
    def inference(self, sd, prompt, n_samples, seed, ddim_steps=1, device='cuda', fixed_code=True, n_rows=0, **kwargs):
        back = [{
            "image": None,
            "text": '',
        }]
        try:
            img = ''
            s_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            path_for_save = f'out_put/{s_time}'
            os.makedirs(path_for_save, exist_ok=True)  #
            outpath = path_for_save
            grid_count = len(os.listdir(outpath)) - 1

            if seed is None:
                seed = randint(0, 1000000)
            seed_everything(seed)

            # # Logging
            # logger(vars(opt), log_csv="logs/txt2img_logs.csv")

            li, lo = [], []
            for key, value in sd.items():
                sp = key.split(".")
                if (sp[0]) == "model":
                    if "input_blocks" in sp:
                        li.append(key)
                    elif "middle_block" in sp:
                        li.append(key)
                    elif "time_embed" in sp:
                        li.append(key)
                    else:
                        lo.append(key)
            for key in li:
                sd["model1." + key[6:]] = sd.pop(key)
            for key in lo:
                sd["model2." + key[6:]] = sd.pop(key)

            config = OmegaConf.load(f"v1-inference.yaml")

            model = instantiate_from_config(config.modelUNet)
            _, _ = model.load_state_dict(sd, strict=False)
            model.eval()
            model.unet_bs = 1
            model.cdevice = 'cuda'
            model.turbo = True

            modelCS = instantiate_from_config(config.modelCondStage)
            _, _ = modelCS.load_state_dict(sd, strict=False)
            modelCS.eval()
            modelCS.cond_stage_model.device = device

            modelFS = instantiate_from_config(config.modelFirstStage)
            _, _ = modelFS.load_state_dict(sd, strict=False)
            modelFS.eval()
            del sd

            if device != "cpu":
                model.half()
                modelCS.half()

            start_code = None
            if fixed_code:
                start_code = torch.randn([n_samples, 4, 512 // 8, 512 // 8], device=device)

            batch_size = n_samples
            n_rows = n_rows if n_rows > 0 else batch_size
            assert prompt is not None
            print(f"Using prompt: {prompt}")
            data = [batch_size * [prompt]]

            if device != "cpu":
                precision_scope = autocast
            else:
                precision_scope = nullcontext

            seeds = ""
            with torch.no_grad():
                all_samples = list()
                for n in trange(1, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):

                        sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                        os.makedirs(sample_path, exist_ok=True)
                        base_count = len(os.listdir(sample_path))

                        with precision_scope("cuda"):
                            modelCS.to(device)
                            uc = None
                            if 7.5 != 1.0:
                                uc = modelCS.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)

                            subprompts, weights = split_weighted_subprompts(prompts[0])
                            if len(subprompts) > 1:
                                c = torch.zeros_like(uc)
                                totalWeight = sum(weights)
                                # normalize each "sub prompt" and add it
                                for i in range(len(subprompts)):
                                    weight = weights[i]
                                    # if not skip_normalize:
                                    weight = weight / totalWeight
                                    c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                            else:
                                c = modelCS.get_learned_conditioning(prompts)

                            shape = [n_samples, 4, 512 // 8, 512 // 8]

                            if device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                modelCS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)

                            samples_ddim = model.sample(
                                S=ddim_steps,
                                conditioning=c,
                                seed=seed,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=7.5,
                                unconditional_conditioning=uc,
                                eta=0.0,
                                x_T=start_code,
                                sampler='plms',
                            )

                            modelFS.to(device)

                            print(samples_ddim.shape)
                            print("saving images")
                            for i in range(batch_size):
                                x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                seeds += str(seed) + ","
                                seed += 1
                                base_count += 1

                            if device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                modelFS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)
                            del samples_ddim
                            print("memory_final = ", torch.cuda.memory_allocated() / 1e6)
            back = [{
                "image": img,
                "text": prompt,
            }]
            return back
        except Exception as ex:
            print(ex)
            back[0]['text'] = f'出现错误，请联系开发人处理{str(ex)}'
            return back


model = Txt2Img_Model(init_shell=False)
model.load_model()
# result = model.inference(arg1='测试输入文本',arg2='test.jpg')  # 测试
# print(result)

# 启动服务
server = Server(model=model)
server.web_examples.append({
    "prompt": 'a photograph of an astronaut riding a horse',
    "ddim_steps": 50,
    "n_samples": 1,
    "seed": None
})
server.server(port=8080)
