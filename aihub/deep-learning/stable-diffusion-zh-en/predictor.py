# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import numpy as np
import torch
import torch.nn.functional as F
from flagai.model.predictor.utils import viterbi_decode, decode_labels, bert_beamsearch,\
    t5_random_sample, gpt_random_sample, \
    t5_beamsearch, gpt_beamsearch, bert_random_sample, glm_beamsearch, glm_random_sample, cpm_beamsearch
from typing import List, Union, Dict, Tuple, Any
from flagai.model.predictor.gpt import gpt_random_sample_use_cache
from flagai.model.mm.Sampler import DDIMSampler, PLMSSampler
import os
from PIL import Image
from tqdm import trange, tqdm
import time
from contextlib import contextmanager, nullcontext
from einops import rearrange

class Predictor:

    def __init__(self, model, tokenizer=None):
        """
        Args:
            model: The model loaded by the AutoLoader class.
            tokenizer: The tokenizer loaded by the AutoLoader class.
        Examples::
            # Define the predictor
            >>> predicter = Predictor(model=model, tokenizer=tokenizer)

        """
        self.tokenizer = tokenizer
        word2idx = None
        if getattr(self.tokenizer, "get_vocab", None) is not None:
            word2idx = self.tokenizer.get_vocab()

        if getattr(self.tokenizer, "token_end_id", None) is None:
            if word2idx is not None:
                if word2idx.get("[SEP]", None) is not None:
                    setattr(self.tokenizer, "token_end_id", word2idx["[SEP]"])
                elif word2idx.get("</s>", None) is not None:
                    setattr(self.tokenizer, "token_end_id", word2idx["</s>"])
                else:
                    setattr(self.tokenizer, "token_end_id", 1)

        if getattr(self.tokenizer, "token_start_id", None) is None:
            if word2idx is not None:
                if word2idx.get("[CLS]", None) is not None:
                    setattr(self.tokenizer, "token_start_id",
                            word2idx["[CLS]"])
                elif word2idx.get("<s>", None) is not None:
                    setattr(self.tokenizer, "token_start_id", word2idx["<s>"])
                else:
                    setattr(self.tokenizer, "token_start_id", 0)

        if getattr(self.tokenizer, "token_unk_id", None) is None:
            if word2idx is not None:
                if word2idx.get("[UNK]", None) is not None:
                    setattr(self.tokenizer, "token_unk_id", word2idx["[UNK]"])
                elif word2idx.get("<unk>", None) is not None:
                    setattr(self.tokenizer, "token_unk_id", word2idx["<unk>"])
                else:
                    setattr(self.tokenizer, "token_unk_id", 0)

        if getattr(self.tokenizer, "token_pad_id", None) is None:
            if word2idx is not None:
                if word2idx.get("[PAD]", None) is not None:
                    setattr(self.tokenizer, "token_pad_id", word2idx["[PAD]"])
                elif word2idx.get("<pad>", None) is not None:
                    setattr(self.tokenizer, "token_pad_id", word2idx["<pad>"])
                else:
                    setattr(self.tokenizer, "token_pad_id", 0)

        self.model = model
        self.model.eval()
        self.class_name = type(model).__name__
        #self.word2idx = self.tokenizer.vocab #to do: GLMLargeChTokenizer add attribute '_vocab'

    def predict_embedding(self, text, maxlen=256):
        device = next(self.model.parameters()).device
        tokenizer_out = self.tokenizer.encode_plus(text,
                                                   max_length=maxlen,
                                                   truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            score = self.model(**{
                "input_ids": input_ids,
                "segment_ids": token_type_ids
            })["logits"].cpu().mean(1)[0]

        return score

    def predict_cls_classifier(self,
                               text: Union[str, List[str]],
                               maxlen: int = 512) -> int:
        """
        Args:
           text: The input. text-pair for semantic matching and text for text classification.
           maxlen: The max length of input.
        """
        device = next(self.model.parameters()).device
        if type(text) is str:
            tokenizer_out = self.tokenizer.encode_plus(text,
                                                       max_length=maxlen,
                                                       truncation=True)
        else:
            assert len(text) == 2
            tokenizer_out = self.tokenizer.encode_plus(text[0],
                                                       text[1],
                                                       max_length=maxlen,
                                                       truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            score = self.model(**{
                "input_ids": input_ids,
                "segment_ids": token_type_ids
            })["logits"].cpu()
        score = score.argmax(dim=-1)
        return score.item()

    def predict_masklm(self, text: str, maxlen: int = 512) -> str:
        """
        Args:
          text: The input text.
          maxlen: The max length of input.
        """
        device = next(self.model.parameters()).device
        tokenizer_out = self.tokenizer.encode_plus(text,
                                                   max_length=maxlen,
                                                   truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            score = self.model(**{
                "input_ids": input_ids,
                "segment_ids": token_type_ids
            })["logits"].cpu()
        score = score.argmax(dim=-1).numpy()[0]
        return self.tokenizer.decode(score)

    def predict_ner(self,
                    text: str,
                    target: List[str],
                    maxlen: int = 256,
                    add_spatial_token=False) -> List[Tuple[int, int, str]]:
        """
        Args:
          text: The input text.
          target: The all text of labels, for example: ["B-LOC", "I-LOC", ...]
          maxlen: The max length of input.
        """
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text,
                                    maxlen=maxlen,
                                    add_spatial_tokens=add_spatial_token)

        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.text_tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        trans = model.state_dict().get("crf_layer.trans", None)
        if trans is not None:
            # crf
            trans = trans.cpu()
            with torch.no_grad():
                out = model(**{"input_ids": token_ids})["logits"][0].cpu()
            labels = viterbi_decode(out, trans)
            entities = decode_labels(labels, target)
            return [(mapping[w[0]][0], mapping[w[-1]][-1], l)
                    for w, l in entities if mapping[w[0]] and mapping[w[-1]]]

        elif getattr(model, "gp", None) is not None:
            entities = []
            with torch.no_grad():
                scores = model(
                    **{"input_ids": token_ids})["logits"].cpu().numpy()[0]
            # global pointer
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf
            for pos_t, start, end in zip(*np.where(scores > 0)):
                if mapping[start] and mapping[end]:
                    entities.append(
                        (mapping[start][0], mapping[end][-1], target[pos_t]))
            return entities
        else:
            with torch.no_grad():
                scores = model(**{"input_ids": token_ids})["logits"].cpu()[0]
            labels = scores.argmax(dim=-1)
            entities = decode_labels(labels, target)
            return [(mapping[w[0]][0], mapping[w[-1]][-1], l)
                    for w, l in entities if mapping[w[0]] and mapping[w[-1]]]

    def predict_generate_beamsearch(self,
                                    text: str,
                                    input_max_length: int = 256,
                                    out_max_length: int = 100,
                                    beam_size: int = 1) -> str:
        """
        Args:
          text: The input text.
          input_max_length: The max length of input text.
          out_max_length: The max length of output text.
          beam_size: The beam size.
        """
        self.model.eval()
        if "glm" in self.class_name.lower():
            #assert "seq2seq" in self.class_name.lower(), "this function only support seq2seq task"
            return glm_beamsearch(self.model, self.tokenizer, text,
                                  out_max_length, beam_size)
        if "bert" in self.class_name.lower():
            assert "seq2seq" in self.class_name.lower(
            ), "this function only support seq2seq task"
            return bert_beamsearch(self.model,
                                   self.tokenizer,
                                   text,
                                   input_max_length=input_max_length,
                                   out_max_length=out_max_length,
                                   beam_size=beam_size)
        elif "t5" in self.class_name.lower():
            return t5_beamsearch(self.model,
                                 self.tokenizer,
                                 text,
                                 input_max_length=input_max_length,
                                 out_max_length=out_max_length,
                                 beam_size=beam_size)

        elif "gpt" in self.class_name.lower():
            return gpt_beamsearch(self.model,
                                  self.tokenizer,
                                  text,
                                  input_max_length=input_max_length,
                                  out_max_length=out_max_length,
                                  beam_size=beam_size)
        elif "cpm" in self.class_name.lower():
            print('self.tokenizer', self.tokenizer)
            return cpm_beamsearch(self.model,
                                  self.tokenizer,
                                  text,
                                  input_max_length=input_max_length,
                                  out_max_length=out_max_length,
                                  beam_size=beam_size)
        else:
            print("Unsupported decoding mode")
            import os
            os._exit(0)

    def predict_generate_randomsample(self,
                                      text: str,
                                      input_max_length: int = 256,
                                      out_max_length: int = 200,
                                      top_k: int = 30,
                                      top_p: float = 1.0,
                                      repetition_penalty: float = 1.0,
                                      temperature: float = 1.0):
        """
        Args:
        text: The input text.
        input_max_length: The max length of input text.
        out_max_length: The max length of output text.
        top_k: keep only top k tokens with highest probability (top-k filtering).
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).(http://arxiv.org/abs/1904.09751)
        repetition_penalty: avoid the repetition out. (https://arxiv.org/pdf/1909.05858.pdf)
        temperature: normalization the score.
        """
        device = next(self.model.parameters()).device
        if "t5" in self.class_name.lower():
            return t5_random_sample(self.model, self.tokenizer, text,
                                    input_max_length, out_max_length, top_k,
                                    top_p, repetition_penalty, temperature,
                                    device)

        elif "gpt" in self.class_name.lower(
        ) or "opt" in self.class_name.lower():
            return gpt_random_sample_use_cache(self.model, self.tokenizer,
                                               text, input_max_length,
                                               out_max_length, top_k, top_p,
                                               repetition_penalty, temperature,
                                               device)
        elif "glm" in self.class_name.lower():
            return glm_random_sample(self.model, self.tokenizer, text,
                                     out_max_length, top_k, top_p,
                                     repetition_penalty, temperature, device)

        elif "bert" in self.class_name.lower():
            assert "seq2seq" in self.class_name.lower(
            ), "this function only support seq2seq task"
            return bert_random_sample(self.model, self.tokenizer, text,
                                      input_max_length, out_max_length, top_k,
                                      top_p, repetition_penalty, temperature,
                                      device)

        else:
            print("Unsupported decoding mode")
            import os
            os._exit(0)

    def predict_generate_images(self,
                                prompt: str,
                                pic_name: str,
                                outpath: str = "AltDiffusionOutputs",
                                n_samples: int = 4,
                                n_rows: int = 0,
                                skip_grid: bool = False,
                                skip_save: bool = False,
                                ddim_steps: int = 50,
                                n_iter: int = 1,
                                plms: bool = False,
                                fixed_code: bool = False,
                                ddim_eta: float = 0.0,
                                H: int = 512,
                                W: int = 512,
                                C: int = 4,
                                f: int = 8,
                                scale: float = 7.5,
                                from_file: str = None,
                                seed: int = 34234,
                                ):
        from torchvision.utils import make_grid
        from pytorch_lightning import seed_everything
        from flagai.model.predictor.utils import chunk, check_safety, get_safety_checker
        safety_checker, safety_feature_extractor = get_safety_checker()
        """
        Args:
        prompt: the prompt text
        out_path: the output path
        n_samples: how many images to be generated
        skip_grid: not to grid images
        skip_save: do not save images
        ddim_step: number of steps in ddim model
        n_iter: number of iterations
        plms: use PLMSSampler
        fixed_code: sampled from a initial start code
        seed: Random seed
        H: height of image
        W: width of image
        C: channels of images, 4 for colored images
        """
        seed_everything(seed)
        dir_list = []
        assert "diffusion" in self.class_name.lower()
        device = next(self.model.parameters()).device
        if plms:
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)

        os.makedirs(outpath, exist_ok=True)

        batch_size = n_samples
        n_rows = n_rows if n_rows > 0 else batch_size
        if not from_file:
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {from_file}")
            with open(from_file, "r") as f:
                data = f.read().splitlines()
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if fixed_code:
            start_code = torch.randn([n_samples, C, H // f, W // f],
                                     device=device)

        precision_scope = nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(
                                    batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = sampler.sample(
                                S=ddim_steps,
                                conditioning=c,
                                batch_size=n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(
                                samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(
                                0, 2, 3, 1).numpy()

                            # x_checked_image, has_nsfw_concept = check_safety(safety_checker, safety_feature_extractor, x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(
                                x_samples_ddim).permute(0, 3, 1, 2)

                            prompt_count = 0
                            if not skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(
                                        x_sample.cpu().numpy(),
                                        'c h w -> h w c')
                                    img = Image.fromarray(
                                        x_sample.astype(np.uint8))
                                    img.save(
                                        os.path.join(sample_path,
                                                     f"{pic_name}-{base_count:05}.png"))
                                    dir_list.append(f"result/samples/{pic_name}-{base_count:05}.png")
                                    #img.save(os.path.join(sample_path, f"{prompts[prompt_count]}.png"))

                                    base_count += 1
                                    prompt_count = prompt_count % batch_size
                                    prompt_count += 1

                            if not skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(
                            grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        # img = put_watermark(img, wm_encoder)
                        img.save(
                            os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

        print(
            f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
        return dir_list
