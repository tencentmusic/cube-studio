import base64
import io, sys, os
from cubestudio.aihub.model import Model
from cubestudio.aihub.docker import Docker
from cubestudio.aihub.web.server import Server, Field, Field_type

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, 'optimizedSD')))

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
    name = 'stable-diffusers'
    label = '文字转图像'
    description = "输入一串文字描述，可生成相应的图片"
    field = "神经网络"
    scenes = "图像创作"
    status = 'online'
    version = 'v20221022'
    doc = 'https://github.com/CompVis/stable-diffusion'  # 'https://帮助文档的链接地址'
    pic = 'https://images.nightcafe.studio//assets/stable-tile.jpg'  # https://应用描述的缩略图/可以直接使用应用内的图片文件地址
    inference={
        "resource_memory":"0",
        "resource_cpu":"0",
        "resource_gpu":"1"
    }
    # 运行基础环境脚本
    init_shell = 'init.sh'

    inference_inputs = [
        Field(type=Field_type.text, name='prompt', label='输入的文字内容',
              describe='输入的文字内容，暂仅支持英文输入~', default='a photograph of an astronaut riding a horse'),
        Field(type=Field_type.int, name='ddim_steps', label='推理的次数',
              describe='推理进行的次数，推荐20-50次将会得到更接近真实的图片', default=50),
        Field(type=Field_type.int, name='n_samples', label='推理出的图像数量(不支持修改!)',
              describe='结果中所展示的图片数量，数量越多则会导致性能下降', default=1)
    ]

    # 加载模型
    # @pysnooper.snoop()
    def load_model(self):
        self.device = 'cuda'   # cuda
        pl_sd = torch.load('/model.ckpt', map_location="cuda:0")
        self.sd = pl_sd["state_dict"]

        sd = self.sd

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

        self.model = instantiate_from_config(config.modelUNet)
        _, _ = self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.model.unet_bs = 1
        self.model.cdevice =self.device
        self.model.turbo = True

        self.modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = self.modelCS.load_state_dict(sd, strict=False)
        self.modelCS.eval()
        self.modelCS.cond_stage_model.device = self.device

        self.modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = self.modelFS.load_state_dict(sd, strict=False)
        self.modelFS.eval()
        del sd

        if self.device != "cpu":
            self.model.half()
            self.modelCS.half()

    # 推理
    # @pysnooper.snoop()
    def inference(self, prompt, n_samples=1, ddim_steps=50, fixed_code=True, n_rows=0, **kwargs):
        begin_time = datetime.datetime.now()
        back = [{
            "image": None,
            "text": '',
        }]
        try:
            seed = randint(0, 1000000)
            seed_everything(seed)

            img = ''
            s_time = datetime.datetime.now().strftime("%Y%m%d")
            outpath = f'result/{s_time}'
            os.makedirs(outpath, exist_ok=True)  #

            start_code = None
            if fixed_code:
                start_code = torch.randn([n_samples, 4, 512 // 8, 512 // 8], device=self.device)

            batch_size = n_samples
            n_rows = n_rows if n_rows > 0 else batch_size
            assert prompt is not None
            print(f"Using prompt: {prompt}")
            data = [batch_size * [prompt]]

            if self.device != "cpu":
                precision_scope = autocast
            else:
                precision_scope = nullcontext
            image_paths=[]
            seeds = ""
            with torch.no_grad():
                all_samples = list()
                for n in trange(1, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):

                        sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
                        os.makedirs(sample_path, exist_ok=True)
                        base_count = len(os.listdir(sample_path))

                        with precision_scope(self.device):
                            self.modelCS.to(self.device)
                            uc = None
                            if 7.5 != 1.0:
                                uc = self.modelCS.get_learned_conditioning(batch_size * [""])
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
                                    c = torch.add(c, self.modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                            else:
                                c = self.modelCS.get_learned_conditioning(prompts)

                            shape = [n_samples, 4, 512 // 8, 512 // 8]

                            if self.device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                self.modelCS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)

                            samples_ddim = self.model.sample(
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

                            self.modelFS.to(self.device)

                            print(samples_ddim.shape)
                            print("saving images")
                            for i in range(batch_size):
                                x_samples_ddim = self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                save_path = os.path.join(sample_path,str(i)+".jpg")
                                img.save(save_path)
                                image_paths.append(save_path)
                                seeds += str(seed) + ","
                                seed += 1
                                base_count += 1

                            if self.device != "cpu":
                                mem = torch.cuda.memory_allocated() / 1e6
                                self.modelFS.to("cpu")
                                while torch.cuda.memory_allocated() / 1e6 >= mem:
                                    time.sleep(1)
                            del samples_ddim
                            print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

            back = [
                {
                    "image": img_path
                } for img_path in image_paths
            ]
            print('花费时长:',(datetime.datetime.now()-begin_time).seconds)
            return back
        except Exception as ex:
            print(ex)
            back[0]['text'] = f'出现错误，请联系开发人处理{str(ex)}'
            return back


model = Txt2Img_Model()
model.load_model()
result = model.inference(prompt='a photograph of an astronaut riding a horse',device='cpu')  # 测试
print(result)

# 启动服务
server = Server(model=model)
server.web_examples.append({
    "prompt": 'a photograph of an astronaut riding a horse',
    "ddim_steps": 50,
    "n_samples": 1
})
server.server(port=8080)
