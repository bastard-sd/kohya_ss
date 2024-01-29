import importlib
import argparse
import gc
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml
from torchvision import transforms

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from library.ipex import ipex_init

        ipex_init()
except Exception:
    pass
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import model_util

import library.train_util as train_util
from library.train_util import (
    DreamBoothDataset,
)
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)
import signal
import sys
from sdxl_gen_img import get_weighted_text_embeddings as get_weighted_sdxl_text_embeddings
import clip
import torch.nn as nn
import pytorch_lightning as pl
import safetensors.torch
import re
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
from transformers import CLIPModel, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from library import sdxl_model_util
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision
from typing import Tuple, List
import numpy as np
import pandas as pd
from diffusers import AutoencoderTiny

import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
from blip import BLIP_Pretrain, load_checkpoint
import json
import io
import tempfile
from inspect import signature

DEBUG = True

def do(func, *args, target=None, label=None, **kwargs):
    if DEBUG:
        label_str = f"{label}: " if label is not None else ""
        if hasattr(target, 'size'):
            print(f"{label_str}{target.size()}")
        elif hasattr(target, '__len__'):
            print(f"{label_str}{len(target)}")
        result = func(*args, **kwargs)
        if hasattr(result, 'size'):
            print(f"{label_str} size: {result.size()}")
        elif hasattr(result, '__len__'):
            print(f"{label_str} len: {len(result)}")
        else:
            print(f"{label_str} {result}")
        return result
    else:
        return func(*args, **kwargs)



# for network_module in network.text_encoder_loras:
#     random_dropout = random.uniform(0, args.network_dropout)
#     network_module.dropout = random_dropout

# def preprocess(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         lambda image: image.convert("RGB"),
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])


def positive_classifier_mse(original_scores, denoised_scores, power=2, min_original_score=-100, negative_loss_scale=0.5):
    # Check if the tensors have more than one dimension and flatten if needed
    if original_scores.ndim > 1:
        original_scores = original_scores.view(original_scores.size(0), -1)
    if denoised_scores.ndim > 1:
        denoised_scores = denoised_scores.view(denoised_scores.size(0), -1)

    # Compute the positive loss term only where the mask is True
    positive_loss = torch.maximum(torch.zeros_like(original_scores), original_scores - denoised_scores) ** power
    negative_loss = torch.maximum(torch.zeros_like(original_scores), denoised_scores - original_scores) ** power
    
    # Combine the loss terms, applying the negative_loss_scale only to the negative loss
    loss = positive_loss - negative_loss_scale * negative_loss

    # Only apply loss if it met a minimum criteria
    loss = torch.where(original_scores > min_original_score, loss, torch.zeros_like(loss))

    # print(original_scores.shape)
    # print(denoised_scores.shape)
    # print(loss.shape)
    # print(loss)

    # Take the mean (or sum) of the loss across all dimensions except the batch dimension
    if original_scores.ndim > 1:
        #loss = loss.mean(dim=1, keepdim=True)
        loss = loss.mean(dim=1)

    return loss

# For BAD scores we simply swap the order
def negative_classifier_mse(original_scores, denoised_scores):
    return positive_classifier_mse(denoised_scores, original_scores)     


class TextualInversionEmbed():
    def __init__(self, embed_path):
        embed = self._load_embedding(embed_path)
        self.embedding = self._process_embedding(embed)

    def _load_embedding(self, embed_path):
        if embed_path.endswith('.safetensors'):
            return safetensors.torch.load_file(embed_path, device="cpu")
        else:  # Assuming .pt/.bin file
            return torch.load(embed_path, map_location="cpu")

    def _process_embedding(self, embed):
        if 'string_to_param' in embed:
            embed_out = next(iter(embed['string_to_param'].values()))
        elif isinstance(embed, list):
            embed_out = self._aggregate_list_embeddings(embed)
        else:
            embed_out = next(iter(embed.values()))
            
        # normalized_embed = embed_out / embed_out.norm(dim=-1, keepdim=True)
        # self._check_embedding_dimension(embed_out)
        aggregated_tensor = embed_out.squeeze().mean()
        return aggregated_tensor

    def _aggregate_list_embeddings(self, embed_list):
        out_list = [t.reshape(-1, t.shape[-1]) for embed_dict in embed_list 
                    for t in embed_dict.values() if t.shape[-1] == 768]
        return torch.cat(out_list, dim=0)

    def _check_embedding_dimension(self, embed_tensor):
        expected_emb_dim = text_encoders[0].get_input_embeddings().weight.shape[-1]
        if expected_emb_dim != embed_tensor.shape[-1]:
            raise ValueError(f"Loaded embeddings are of incorrect shape. Expected {expected_emb_dim}, but are {embed_tensor.shape[-1]}")

class AestheticModelBase(pl.LightningModule):
    def __init__(self):
        super(AestheticModelBase, self).__init__()  
        self.input_size = 768
        self.layers = None  # To be defined in subclasses

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        self.eval()
        self.half()

    def forward(self, x):
        if self.layers is None:
            raise NotImplementedError("Layers must be defined in subclass")
        return self.layers(x)

class AestheticModelReLU(AestheticModelBase):
    def __init__(self, model_path):
        super(AestheticModelReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Dropout(0.01),
            nn.Linear(16, 1),
        )
        self.load_model(model_path)

class AestheticModelLinear(AestheticModelBase):
    def __init__(self, model_path):
        super(AestheticModelLinear, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),  
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Dropout(0.01),
            nn.Linear(16, 1),
        )
        self.load_model(model_path)

class AestheticModelLinearAlt(AestheticModelBase):
    def __init__(self, input_size, model_path=None):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0)
                
        if model_path:
            self.load_model(model_path)

class ImageRewardModel(pl.LightningModule):
    def __init__(self, aes_model, config, device):
        super().__init__()
        self.blip = BLIP_Pretrain(image_size=224, vit='large', config=config)
        self.mlp = AestheticModelLinearAlt(768)
        
        state_dict = torch.load(aes_model, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        self.to(device)
        
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

















    # def add_image_rewards(self):
    #     model_path = r".\0_bastard\other\ImageReward.pt"
    #     config_obj = {
    #         "architectures": [
    #             "BertModel"
    #         ],
    #         "attention_probs_dropout_prob": 0.1,
    #         "hidden_act": "gelu",
    #         "hidden_dropout_prob": 0.1,
    #         "hidden_size": 768,
    #         "initializer_range": 0.02,
    #         "intermediate_size": 3072,
    #         "layer_norm_eps": 1e-12,
    #         "max_position_embeddings": 512,
    #         "model_type": "bert",
    #         "num_attention_heads": 12,
    #         "num_hidden_layers": 12,
    #         "pad_token_id": 0,
    #         "type_vocab_size": 2,
    #         "vocab_size": 30524,
    #         "encoder_width": 768,
    #         "add_cross_attention": True   
    #     }
  
    #     # Write config to a temporary file
    #     with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp:
    #         json.dump(config_obj, tmp)
    #         config_path = tmp.name

    #     self.model = ImageRewardModel(
    #         model_path,
    #         config_path,
    #         device
    #     )



# class CLIPScore(nn.Module):
#     def __init__(self, pathname, device='cpu'):
#         super().__init__()
#         self.device = device
#         self.clip_model, self.preprocess = clip.load(pathname, device=self.device, jit=False)

#         if device == "cpu":
#             self.clip_model.float()
#         else:
#             clip.model.convert_weights(
#                 self.clip_model)  # Actually this line is unnecessary since clip by default already on float16

#         # have clip.logit_scale require no grad.
#         self.clip_model.logit_scale.requires_grad_(False)

#     def score(self, prompt, image):

#         if (type(image).__name__ == 'list'):
#             _, rewards = self.inference_rank(prompt, image)
#             return rewards

#         # text encode
#         text = clip.tokenize(prompt, truncate=True).to(self.device)
#         txt_features = F.normalize(self.clip_model.encode_text(text))

#         # image encode
#         if isinstance(image, Image.Image):
#             pil_image = image
#         elif isinstance(image, str):
#             if os.path.isfile(image):
#                 pil_image = Image.open(image)
#         image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
#         image_features = F.normalize(self.clip_model.encode_image(image))

#         # score
#         rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)

#         score = rewards.detach().cpu().numpy().item()
#         score += 1
#         score *= 5
#         return score

#     def inference_rank(self, prompt, generations_list):

#         text = clip.tokenize(prompt, truncate=True).to(self.device)
#         txt_feature = F.normalize(self.clip_model.encode_text(text))

#         txt_set = []
#         img_set = []
#         for generations in generations_list:
#             # image encode
#             img_path = generations
#             pil_image = Image.open(img_path)
#             image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
#             image_features = F.normalize(self.clip_model.encode_image(image))
#             img_set.append(image_features)
#             txt_set.append(txt_feature)

#         txt_features = torch.cat(txt_set, 0).float()  # [image_num, feature_dim]
#         img_features = torch.cat(img_set, 0).float()  # [image_num, feature_dim]
#         rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
#         rewards = torch.squeeze(rewards)
#         _, rank = torch.sort(rewards, dim=0, descending=True)
#         _, indices = torch.sort(rank, dim=0)
#         indices = indices + 1

#         return indices.detach().cpu().numpy().tolist(), rewards.detach().cpu().numpy().tolist()

#     def features(self, prompt, image, aes_type='v2'):

#         # text encode
#         text = clip.tokenize(prompt, truncate=True).to(self.device)
#         txt_features = F.normalize(self.clip_model.encode_text(text))

#         # image encode
#         if isinstance(image, Image.Image):
#             pil_image = image
#         elif isinstance(image, str):
#             if os.path.isfile(image):
#                 pil_image = Image.open(image)
#         image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
#         image_features = F.normalize(self.clip_model.encode_image(image))

#         return txt_features, image_features


# class ClipDiscriminator(BaseDiscriminator):
#     def __init__(self, manager, name, weight, loss_func, input_type='embedding'):
#         self.manager = manager
#         self.name = name
#         self.weight = weight
#         self.loss_func = loss_func
#         self.input_type = input_type
#         self.device = manager.device
#         #manager.add_discriminator(name, self)
        
#     def __init__(self, manager, name, weight, loss_func):
#         super().__init__(manager, name, weight, loss_func)
        
#     def compute_scores(self, embeddings):
#         return 5 * torch.nn.functional.cosine_similarity(embeddings, self.embedding, dim=-1)









class BaseDiscriminator:
    def __init__(self, manager, name, weight, loss_func, input_type='embedding'):
        self.manager = manager
        self.name = name
        self.weight = weight
        self.loss_func = loss_func
        self.input_type = input_type
        self.device = manager.device
        #manager.add_discriminator(name, self)

    def compute_loss(self, original_scores, denoised_scores):
        return self.loss_func(original_scores, denoised_scores)

class BaseCosineSimilarityDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func):
        super().__init__(manager, name, weight, loss_func)

    def compute_scores(self, embeddings):
        return 5 * torch.nn.functional.cosine_similarity(embeddings, self.embedding, dim=-1)

class TextDiscriminator(BaseCosineSimilarityDiscriminator):
    def __init__(self, manager, name, weight, loss_func, prompt):
        super().__init__(manager, name, weight, loss_func)
        self.embedding = self._get_text_embedding(prompt).to(manager.device)

    def _get_text_embedding(self, prompt):
        with torch.no_grad():
            clip_model, _ = self.manager.vit_model
            text_input = clip.tokenize(prompt).to(self.device)
            text_embedding = clip_model.encode_text(text_input)
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True)

class PromptDiscriminator(BaseCosineSimilarityDiscriminator):
    def __init__(self, manager, name, weight, loss_func, prompt):
        super().__init__(manager, name, weight, loss_func)
        
        token_replacements_list = []
        for _ in range(len(manager.text_encoders)):
            token_replacements_list.append({})
        
        self.embedding = self._get_weighted_text_embeddings(
            manager.is_sdxl,
            tokenizers=manager.tokenizers,
            text_encoders=manager.text_encoders,
            prompt=prompt,
            token_replacements_list=token_replacements_list,
            device=manager.device
        ).to(manager.device)



    
    def add_token_replacement(self, token_replacements_list, text_encoder_index, target_token_id, rep_token_ids):
        token_replacements_list[text_encoder_index][target_token_id] = rep_token_ids


    def get_token_replacer(self, tokenizers, tokenizer, token_replacements_list=[]):
        tokenizer_index = tokenizers.index(tokenizer)
        token_replacements = []
        if len(token_replacements_list) > 0:
            token_replacements = token_replacements_list[tokenizer_index]

        def replace_tokens(tokens):
            # print("replace_tokens", tokens, "=>", token_replacements)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            new_tokens = []
            for token in tokens:
                if token in token_replacements:
                    replacement = token_replacements[token]
                    new_tokens.extend(replacement)
                else:
                    new_tokens.append(token)
            return new_tokens

        return replace_tokens


    def _get_weighted_text_embeddings(
            self,
            is_sdxl,
            tokenizers=[],
            text_encoders=[],
            prompt='',
            token_replacements_list=[],
            max_embeddings_multiples=1,
            clip_skip=1,
            device='cpu',
            **kwargs,
        ):
        if is_sdxl:
            tes_text_embs = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                token_replacer = self.get_token_replacer(tokenizers, tokenizer, token_replacements_list)
                text_embeddings, _, _, _, _ = get_weighted_sdxl_text_embeddings(
                    tokenizer,
                    text_encoder,
                    prompt=prompt,
                    max_embeddings_multiples=max_embeddings_multiples,
                    clip_skip=clip_skip,
                    token_replacer=token_replacer,
                    device=device,
                    **kwargs,
                )
                tes_text_embs.append(text_embeddings)

            # concat text encoder outputs
            text_embeddings = tes_text_embs[0]
            for i in range(1, len(tes_text_embs)):
                text_embeddings = torch.cat([text_embeddings, tes_text_embs[i]], dim=2)  # n,77,2048

        else:
            text_embeddings = get_weighted_text_embeddings(
                tokenizers[0],
                text_encoders[0],
                prompt,
                device,
                max_embeddings_multiples=max_embeddings_multiples, # 1 * 75 token max length, I think 1 is necessary for compatibility with loaded embeddings.
                clip_skip=clip_skip, # Clip Skip
            )
            
        processed_embedding = text_embeddings[0].squeeze().mean()
        return processed_embedding

class PretrainedEmbeddingDiscriminator(BaseCosineSimilarityDiscriminator):
    def __init__(self, manager, name, weight, loss_func, model):
        super().__init__(manager, name, weight, loss_func)
        self.embedding = model.embedding.to(manager.device)

class AestheticDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, model):
        super().__init__(manager, name, weight, loss_func)
        self.model = model.to(self.device)

    def compute_scores(self, embeddings):
        print('embeddings.size()')
        print(embeddings.size())
        embeddings = self.model(embeddings).squeeze() / 10
        print('embeddings.size()')
        print(embeddings.size())
        print('embeddings')
        print(embeddings)
        return embeddings

class ClassifierDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, model):
        super().__init__(manager, name, weight, loss_func)
        self.model = model.to(self.device)

    def compute_scores(self, embeddings):
        return self.model(embeddings).squeeze()

class FunctionDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, score_func, input_type='noise'):
        super().__init__(manager, name, weight, loss_func, input_type=input_type)
        self.compute_scores = score_func

class ImageRewardDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, model, config, input_type):
        super().__init__(manager, name, weight, loss_func, input_type)
        self.model = ImageRewardModel(model, config, manager.device)

    def compute_scores(self, image_preprocessed, captions):
        text_inputs = do(lambda: [self.model.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=225, return_tensors="pt").to(self.device) for prompt in captions], label='text_inputs')
        image_preprocessed = image_preprocessed.to(self.device)
        image_embeds = do(lambda: self.model.blip.visual_encoder(image_preprocessed), target=image_preprocessed, label='image_embeds')
        txt_features_set = []
        
        for i, text_input in enumerate(text_inputs):
            img_embed = do(lambda: image_embeds[i, :, :].unsqueeze(0), label='img_embed')
            sequence_length = do(lambda: text_input.input_ids.size(1), label='sequence_length')
            encoder_sequence_length = do(lambda: img_embed.size(1), label='encoder_sequence_length')
            image_atts = do(lambda: torch.ones((1, sequence_length, encoder_sequence_length), dtype=torch.long).to(self.device), label='image_atts')
            text_output = do(lambda: self.model.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=img_embed,
                encoder_attention_mask=image_atts,
                return_dict=True),
                label='text_output'
            )
            txt_features = do(lambda: text_output.last_hidden_state[:, 0, :].float())
            txt_features_set.append(txt_features)

        # Concatenate text features from all examples
        txt_features_batch = do(lambda: torch.cat(txt_features_set, 0))
        score = do(lambda: self.model.mlp(txt_features_batch))
        score = do(lambda: (score - self.model.mean) / self.model.std)
        score = do(lambda: score.squeeze())
        return score

class BLIPScoreDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, blip_model, config, input_type):
        super().__init__(manager, name, weight, loss_func, input_type)
        blip = BLIP_Pretrain(image_size=224, vit='large', config=config)
        blip, _ = load_checkpoint(blip, blip_model)
        self.blip = blip
        self.blip.to(manager.device)

    def compute_scores(self, image_preprocessed, captions):
        text_inputs = [self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=75, return_tensors="pt").to(self.blip.device) for prompt in captions]
        image_preprocessed = image_preprocessed.to(self.blip.device)
        image_embeds = self.blip.visual_encoder(image_preprocessed)
        txt_set = []
        img_set = []

        for i, text_input in enumerate(text_inputs):
            text_output = self.blip.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
            txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))
            image_feature = F.normalize(self.blip.vision_proj(image_embeds[i, 0, :]), dim=-1)
            image_feature = image_feature.unsqueeze(0)
            img_set.append(image_feature)
            txt_set.append(txt_feature)
            
        txt_features = torch.cat(txt_set, 0).float()
        img_features = torch.cat(img_set, 0).float()
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        score = rewards
        score = score.squeeze()
        return score

class CLIPScoreDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, clip_model, input_type):
        super().__init__(manager, name, weight, loss_func, input_type)
        self.clip_model, self.preprocess = clip.load(clip_model, device=manager.device, jit=False)
        self.clip_model.logit_scale.requires_grad_(False)

    def compute_scores(self, image_preprocessed, captions):
        txt_set = []
        img_set = []
        for i, prompt in enumerate(captions):
            text_input = clip.tokenize(prompt, truncate=True).to(self.device)
            txt_feature = F.normalize(self.clip_model.encode_text(text_input), dim=-1)
            img_feature = F.normalize(self.clip_model.encode_image(image_preprocessed[i].unsqueeze(0)), dim=-1)
            txt_set.append(txt_feature)
            img_set.append(img_feature)
        
        txt_features = torch.cat(txt_set, 0)
        img_features = torch.cat(img_set, 0)
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        scores = rewards
        scores = scores.squeeze()
        return scores

#from skimage.feature import greycomatrix, greycoprops
#from skimage.feature import local_binary_pattern
class NoiseAnalysis:
            

    @staticmethod
    def skewness(tensor):
        # Ensure tensor is in float format for mean and variance computations
        tensor = tensor.float()

        # Initialize a list to store skewness for each item in the batch
        skewnesses = []

        # Calculate skewness for each item in the batch
        for i in range(tensor.size(0)):  # Iterate over the batch dimension
            # Select the i-th batch item (all elements along all dimensions for this item)
            t = tensor[i].view(-1)

            # Calculate the mean and standard deviation of the flattened tensor
            mean = t.mean()
            std = t.std(unbiased=True)

            # Number of observations
            n = t.numel()

            # Compute the third moment (the numerator of the skewness formula)
            third_moment = torch.mean((t - mean) ** 3)

            # Compute skewness using the formula
            skew = third_moment / (std ** 3)

            # unbias skewness
            #skew = skew * (n / ((n - 1) * (n - 2)))

            # Store the skewness in the list
            skewnesses.append(skew)

        # Convert the list of skewnesses to a tensor and scale it (if necessary, as per your original code)
        skewnesses_tensor = torch.tensor(skewnesses, device=tensor.device)

        # def transform_skewness(skewness_tensor):
        #     # Take the sign of each skewness value
        #     signs = skewness_tensor.sign()

        #     # Apply the cubic root transformation to the absolute values
        #     transformed_skewness = torch.abs(skewness_tensor) ** (1/3)

        #     # Reapply the signs to preserve the direction of skewness
        #     transformed_skewness *= signs

        #     return transformed_skewness
        # skewnesses_tensor = transform_skewness(skewnesses_tensor)

        return skewnesses_tensor

    @staticmethod
    def kurtosis(tensor):
        # Ensure tensor is in float format for mean and variance computations
        tensor = tensor.float()
        
        # Initialize a list to store kurtosis for each item in the batch
        kurtoses = []
        
        # Calculate kurtosis for each item in the batch
        for i in range(tensor.size(0)):  # Iterate over the batch dimension
            # Select the i-th batch item (all elements along all dimensions for this item)
            t = tensor[i].view(-1)
            
            # Calculate the mean and standard deviation of the flattened tensor
            mean = t.mean()
            std = t.std(unbiased=True)
            
            # Compute the fourth moment (the numerator of the kurtosis formula)
            fourth_moment = torch.mean((t - mean) ** 4)
            
            # Compute the kurtosis for this item
            kurt = fourth_moment / (std ** 4)
            
            # Adjust for excess kurtosis
            excess_kurtosis = kurt - 3
            
            # Store the excess kurtosis in the list
            kurtoses.append(excess_kurtosis)
        
        # Convert the list of kurtoses to a tensor and scale it (if necessary, as per your original code)
        kurtoses_tensor = torch.tensor(kurtoses, device=tensor.device)
        
        return kurtoses_tensor

    @staticmethod
    def entropy(tensor):
        entropies = []
        for i in range(tensor.shape[0]):  # Iterate over each item in the batch
            # Compute the standard deviation of the flattened tensor
            mean, std_dev = torch.mean(tensor[i].view(-1)), torch.std(tensor[i].view(-1))
            # Calculate the differential entropy for a normal distribution
            entropy = 0.5 * torch.log2(2 * np.pi * np.e * std_dev**2)
            #std_dev = tensor[i].view(-1).std().cpu().item()
            #entropy = 0.5 * np.log2(2 * np.pi * np.e * std_dev**2)
            entropies.append(entropy)
        return torch.tensor(entropies, device=tensor.device)

    @staticmethod
    def cross_entropy(tensor2, tensor1):
        cross_entropies = []
        with torch.no_grad():
            for i in range(tensor1.shape[0]):  # Iterate over each item in the batch
                # Compute mean and standard deviation of the flattened tensors
                mean1, std_dev1 = torch.mean(tensor1[i].view(-1)), torch.std(tensor1[i].view(-1))
                mean2, std_dev2 = torch.mean(tensor2[i].view(-1)), torch.std(tensor2[i].view(-1))
                
                # Calculate the differential entropy for the true distribution
                #entropy1 = 0.5 * torch.log2(2 * np.pi * np.e * std_dev1**2)
                #entropy2 = 0.5 * torch.log2(2 * np.pi * np.e * std_dev2**2)
                
                # Calculate the cross-entropy between the two distributions
                kl_div = 0.5 * (torch.log2(std_dev2**2 / std_dev1**2) +
                                (std_dev1**2 + (mean1 - mean2)**2) / std_dev2**2 -
                                1)
                #cross_entropy = entropy1 + kl_div - entropy2
                #cross_entropies.append(cross_entropy.item())  # Convert to Python float for appending
                cross_entropies.append(kl_div)
                
        return torch.tensor(cross_entropies, device=tensor1.device)


    @staticmethod
    def sharpness_tensor(images_tensor):
        if images_tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional [batch_size, channels, height, width]")

        num_channels = images_tensor.shape[1]

        # Define a Laplacian kernel
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]], dtype=images_tensor.dtype, device=images_tensor.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

        sharpness_values_per_channel = []

        for i in range(num_channels):
            # Extract the single channel
            single_channel = images_tensor[:, i:i+1, :, :]

            # Apply the Laplacian filter
            laplacian = torch.nn.functional.conv2d(single_channel, laplacian_kernel, padding=1)

            # Compute the standard deviation for each image in the channel
            channel_sharpness = laplacian.view(laplacian.shape[0], -1).std(dim=1)
            sharpness_values_per_channel.append(channel_sharpness)

        # Combine sharpness values from all channels
        # Here we're averaging the sharpness values, but other methods like max can also be used
        combined_sharpness = torch.stack(sharpness_values_per_channel, dim=1).mean(dim=1)

        return combined_sharpness


class DiscriminatorManager:
    gradient_types = ["noise", "latent", "decode", "embedding"]
    def __init__(self, config_json, device, noise_scheduler, tokenizers, text_encoders, is_sdxl, save_image_steps=10, print_diagnostics=False):
        with open(config_json, 'r') as cfg:
            self.config_json = json.load(cfg)
        
        self.discriminators = {}
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.tokenizers = tokenizers
        # for tokenizer in tokenizers:
        #     tokenizer.to(torch.device(device))
        self.text_encoders = text_encoders
        for text_encoder in text_encoders:
            text_encoder.to(torch.device(device))
        self.vit_model = clip.load("ViT-L/14", device=device)
        self.is_sdxl = is_sdxl
        self.print_diagnostics = print_diagnostics
        self.save_image_steps = save_image_steps
        
        sdxl_model_util.VAE_SCALE_FACTOR = 1.0
        self.vae_scale_factor = 1.0
        
        if is_sdxl:
            print('VAE = SDXL')
            self.vae_model = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to(device)
        else:
            print('VAE = SD15')
            self.vae_model = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)
            
        self.add_aesthetics(self.config_json["aesthetics"])
        self.add_blipscores(self.config_json["blipScores"])
        self.add_clipscores(self.config_json["clipScores"])
        self.add_image_rewards(self.config_json["imageRewards"])
        self.add_functions(self.config_json["functions"])
        self.add_embeddings(self.config_json["embeddings"])
        self.add_embedding_texts(self.config_json["embeddingTexts"])

    def classifier_lambda(self, obj):
        if obj["classifierType"] == "custom":
            match = re.search(r'[-+]?\d*\.\d+|\d+$', obj["classifierExpression"])
            # Use regular expression to find the last numeric sequence (integer or float)
            if not match:
                raise ValueError(f"Invalid classifierExpression: {obj['classifierExpression']}")
            elif "negative_loss_scale" in obj["classifierExpression"]:
                scale_value = float(match.group())
                lmbda = lambda x, y: positive_classifier_mse(x, y, negative_loss_scale=scale_value)
            elif "min_original_score" in obj["classifierExpression"]:
                score_value = int(match.group())
                lmbda = lambda x, y: positive_classifier_mse(x, y, min_original_score=score_value)
            elif "power" in obj["classifierExpression"]:
                power_value = int(match.group())
                lmbda = lambda x, y: positive_classifier_mse(x, y, power=power_value)
        else:
            lmbda = positive_classifier_mse if obj["classifierType"] == 'positive_classifier_mse' else negative_classifier_mse
        
        return lmbda

    def add_aesthetics(self, aesthetics):
        for aesthetic in aesthetics:
            model_type = AestheticModelLinear if aesthetic["modelType"] == 'AestheticModelLinear' else AestheticModelReLU
            self.add_aesthetic(
                aesthetic["name"],
                aesthetic["priority"],
                self.classifier_lambda(aesthetic),
                model_type(aesthetic["filePath"])
            )  
            
    def add_blipscores(self, blipscores):
        config_obj = {
            "architectures": [
                "BertModel"
            ],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30524,
            "encoder_width": 768,
            "add_cross_attention": True   
        }

        for blipscore in blipscores:
            # Write config to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp:
                json.dump(config_obj, tmp)
                tmp_path = tmp.name

            # Now tmp_path is the file path to the temporary file containing the config
            self.add_blipscore(
                blipscore["name"],
                blipscore["priority"],
                self.classifier_lambda(blipscore),
                blipscore["filePath"],
                tmp_path,
                blipscore["input_type"]
            )  
            
    def add_clipscores(self, clipscores):
        for clipscore in clipscores:
            self.add_clipscore(
                clipscore["name"],
                clipscore["priority"],
                self.classifier_lambda(clipscore),
                clipscore["filePath"],
                clipscore["input_type"]
            )
 
    def add_image_rewards(self, imagerewards):
        config_obj = {
            "architectures": [
                "BertModel"
            ],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30524,
            "encoder_width": 768,
            "add_cross_attention": True   
        }
  
        for imagereward in imagerewards:
            # Write config to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp:
                json.dump(config_obj, tmp)
                tmp_path = tmp.name

            # Now tmp_path is the file path to the temporary file containing the config
            self.add_image_reward(
                imagereward["name"],
                imagereward["priority"],
                self.classifier_lambda(imagereward),
                imagereward["filePath"],
                tmp_path,
                imagereward["input_type"]
            )
    
    def add_functions(self, functions):
        for function in functions:
            if "NoiseAnalysis" in function["functionType"]:
                if "sharpness_tensor" in function["functionType"]:
                    function_lambda = NoiseAnalysis.sharpness_tensor
                elif "entropy" in function["functionType"]:
                    function_lambda = NoiseAnalysis.entropy
                elif "kurtosis" in function["functionType"]:
                    function_lambda = NoiseAnalysis.kurtosis
            elif "custom" in function["functionType"]:
                if "Entropy - Kurtosis" in function["functionExpression"]:
                    function_lambda = lambda x: NoiseAnalysis.entropy(x) - NoiseAnalysis.kurtosis(x)
            
            self.add_function(
                function["name"],
                function["priority"],
                self.classifier_lambda(function),
                function_lambda,
                function.get("additionalParameter", None)  # Optional parameter
            )
            
    def add_functions_new(self, functions):
        for function in functions:
            function_lambda = None

            if function["classifierType"] == "custom":
                # Custom logic for custom classifier types
                if "Entropy - Kurtosis" in function["functionExpression"]:
                    function_lambda = lambda x: NoiseAnalysis.entropy(x) - NoiseAnalysis.kurtosis(x)
            else:
                func_components = function["functionType"].split('.')
                if len(func_components) == 2:
                    # Module and function name are specified
                    module_name, func_name = func_components
                    module = importlib.import_module(module_name)
                    function_lambda = getattr(module, func_name, None)
                else:
                    # Global function or single module function
                    function_lambda = globals().get(func_components[0], None)

            if function_lambda:
                self.add_function(
                    function["name"],
                    function["priority"],
                    self.classifier_lambda(function),
                    function_lambda,
                    function.get("additionalParameter", None)  # Optional parameter
                )
                        
    
    def add_embeddings(self, embeddings):
        for embedding in embeddings:
            embedding_model = None
            if "TextualInversionEmbed" in embedding["embeddingType"]:
                embedding_model = TextualInversionEmbed(embedding["filePath"])
            self.add_embedding_pretrained(
                embedding["name"],
                embedding["priority"],
                self.classifier_lambda(embedding),
                embedding_model
            )
            
    def add_embedding_texts(self, embedding_texts):
        for embedding_text in embedding_texts:
            self.add_embedding_text(
                embedding_text["name"],
                embedding_text["priority"],
                self.classifier_lambda(embedding_text),
                embedding_text["text"],
                multi_token=embedding_text["multiToken"] if "multiToken" in embedding_text else False
            )

    def add_aesthetic(self, name, weight, loss_func, model):
        self.discriminators[name] = AestheticDiscriminator(self, name, weight, loss_func, model)

    def add_classifier(self, name, weight, loss_func, model):
        self.discriminators[name] = ClassifierDiscriminator(self, name, weight, loss_func, model)

    def add_embedding_text(self, name, weight, loss_func, prompt, multi_token=False):
        if multi_token:
            self.discriminators[name] = PromptDiscriminator(self, name, weight, loss_func, prompt)
        else:
            self.discriminators[name] = TextDiscriminator(self, name, weight, loss_func, prompt)

    def add_embedding_pretrained(self, name, weight, loss_func, embed_path):
        self.discriminators[name] = PretrainedEmbeddingDiscriminator(self, name, weight, loss_func, embed_path)

    def add_function(self, name, weight, loss_func, score_func, input_type):
        self.discriminators[name] = FunctionDiscriminator(self, name, weight, loss_func, score_func, input_type)

    def add_image_reward(self, name, weight, loss_func, model, config, input_type):
        self.discriminators[name] = ImageRewardDiscriminator(self, name, weight, loss_func, model, config, input_type)

    def add_blipscore(self, name, weight, loss_func, model, config, input_type):
        self.discriminators[name] = BLIPScoreDiscriminator(self, name, weight, loss_func, model, config, input_type)

    def add_clipscore(self, name, weight, loss_func, model, input_type):
        self.discriminators[name] = CLIPScoreDiscriminator(self, name, weight, loss_func, model, input_type)







    def remove_noise(
        #noise_scheduler,
        self,
        noisy_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        noise_scheduler = self.noise_scheduler
        # Make sure alphas_cumprod and timestep have the same device and dtype as noisy_samples
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_samples.device, dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Reverse the noise addition process
        original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
        return original_samples


    def scale_losses(self, loss, timesteps):
        # Since we've converted from noise back to image space, we should use inverse of debiased estimation 
        # to obtain a uniform loss scale across timesteps

        ones = torch.ones_like(timesteps)
        scaling_factors = apply_debiased_estimation(ones, timesteps, self.noise_scheduler)
        scaling_factors = 1.0 / (scaling_factors + 1e-7)
        #scaling_factors *= ones + torch.abs(timesteps - 500) / 500


        # Step 4: Multiply the original losses by the inverse scaling factor
        return loss * scaling_factors

    def apply_discriminator_losses(self, base_loss, timesteps, original_latents, noise, noisy_latents, noise_pred, step, output_name, output_dir, captions):
        
        def compute_discriminator_loss(discriminator, discriminator_name, original, denoised, timesteps, base_loss, captions):
            import contextlib
            @contextlib.contextmanager
            def conditional_grad(input_type):
                if input_type in self.gradient_types:
                    with torch.no_grad():
                        yield
                else:
                    yield

            with torch.no_grad():
                if 'captions' in signature(discriminator.compute_scores).parameters:
                    original_scores = discriminator.compute_scores(original[discriminator.input_type], captions)
                else:
                    original_scores = discriminator.compute_scores(original[discriminator.input_type])
          

            with conditional_grad(discriminator.input_type):       
                if 'captions' in signature(discriminator.compute_scores).parameters:
                    denoised_scores = discriminator.compute_scores(denoised[discriminator.input_type], captions)
                else:
                    denoised_scores = discriminator.compute_scores(denoised[discriminator.input_type])
                discriminator_loss = discriminator.loss_func(original_scores, denoised_scores)

                """
                Q = alphas_cumprod_t
                SNR = Q / (1-Q)
                debias = 1 / SNR^0.5
                debias = sqrt((1-Q)/Q)
                """            
                """            
                ### if discriminator.input_type != 'noise':
                ###     discriminator_loss = self.scale_losses(discriminator_loss, timesteps)
                """
                ### if discriminator.input_type != 'noise':
                ###     discriminator_loss = self.scale_losses(discriminator_loss, timesteps)
                discriminator_loss *= discriminator.weight

                diagnostics = self._accumulate_diagnostics(discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss)
                return discriminator_loss, discriminator, diagnostics

        with torch.no_grad():
            original = {}
            denoised = {}
            
            original["noise"] = noise
            denoised["noise"] = noise_pred

            original["latent"] = original_latents
            with torch.enable_grad():
                denoised["latent"] = self.remove_noise(noisy_latents, noise_pred, timesteps)

            # Decode latents and get images, embeddings
            self._process_batch(original, denoised, step, timesteps, output_name, output_dir)
            
            # Apply each discriminator in parallel
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = []
                for discriminator_name, discriminator in self.discriminators.items():
                    args = (discriminator, discriminator_name, original, denoised, timesteps, base_loss, captions)
                    future = executor.submit(compute_discriminator_loss, *args)
                    futures.append(future)
                
                all_diagnostics = []
                total_batches = 0
                gradient_loss = 0
                ungradient_loss = 0
                for future in futures:
                    discriminator_loss, discriminator, diagnostics = future.result()
                    if discriminator.input_type in self.gradient_types:
                        with torch.enable_grad():
                            if discriminator_loss.ndim == 0:  # Check if tensor is scalar
                                gradient_loss += discriminator_loss
                                total_batches += 1
                            else:
                                gradient_loss += discriminator_loss.sum()
                                total_batches += discriminator_loss.size(0)
                    else:
                        if discriminator_loss.ndim == 0:  # Check if tensor is scalar
                            ungradient_loss += discriminator_loss
                            total_batches += 1
                        else:
                            ungradient_loss += discriminator_loss.sum()
                            total_batches += discriminator_loss.size(0)
                    all_diagnostics.extend(diagnostics)

                # Normalize the losses by the number of batches
                if total_batches > 0:
                    gradient_loss /= total_batches
                    ungradient_loss /= total_batches

            # Pivoting and displaying the data
            if self.print_diagnostics:
                df = pd.DataFrame(all_diagnostics)
                self._pivot_and_display(df)

            # # Compute the scaling factor
            # denominator = base_loss + gradient_loss
            # if base_loss + gradient_loss == 0:
            #     final_loss = base_loss + gradient_loss + ungradient_loss
            # else:
            #     modified_loss_scale = (base_loss + gradient_loss + ungradient_loss) / denominator
            #     final_loss = (base_loss + gradient_loss) * modified_loss_scale.detach()
                        
        return base_loss + gradient_loss + ungradient_loss

    def _accumulate_diagnostics(self, discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss):
        with torch.no_grad():
            diagnostic_data = []

            # Ensure tensors are 2D (batch_size x data)
            original_scores = original_scores.view(-1, 1)
            denoised_scores = denoised_scores.view(-1, 1)
            discriminator_loss = discriminator_loss.view(-1, 1)
            base_loss = base_loss.view(-1, 1)
            timesteps = timesteps.view(-1, 1)

            # Convert to numpy arrays for easier processing
            original_scores_np = original_scores.cpu().numpy()
            denoised_scores_np = denoised_scores.cpu().numpy()
            discriminator_loss_np = discriminator_loss.cpu().numpy()
            base_loss_np = base_loss.cpu().numpy()
            timesteps_np = timesteps.cpu().numpy()

            for i in range(discriminator_loss_np.shape[0]):
                # Check if the scores are simple floats (ndim == 2 and size == 1 for 2nd dim)
                #if original_scores.ndim == 2 and original_scores.size(1) == 1 and denoised_scores.ndim == 2 and denoised_scores.size(1) == 1:
                if discriminator_loss_np.shape[0] == original_scores_np.shape[0]:
                    diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Orig", "Value": original_scores_np[i, 0], "TS": timesteps_np[i, 0]})
                    diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Deno", "Value": denoised_scores_np[i, 0], "TS": timesteps_np[i, 0]})

                # Always add the loss
                diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Loss", "Value": discriminator_loss_np[i, 0], "TS": timesteps_np[i, 0]})

                # Add original latent loss (only once per batch, not per discriminator)
                diagnostic_data.append({"Batch": i, "Discriminator": "latent", "Type": "Loss", "Value": base_loss_np[i, 0], "TS": timesteps_np[i, 0]})

            return diagnostic_data

    def _pivot_and_display(self, df, row_order=['Type', 'Batch', 'TS'], max_row_length=120):
        df = df.drop_duplicates(subset=row_order + ['Discriminator'])

        # Pivot the DataFrame
        pivoted_df = df.pivot_table(index=row_order, columns='Discriminator', values='Value', aggfunc='first')


        # # Check if 'Loss' is in the 'Type' level after pivoting
        # if 'Loss' in pivoted_df.index.get_level_values('Type'):
        #     # Sum only the 'Loss' rows
        #     loss_rows = pivoted_df.xs('Loss', level='Type')
        #     total_loss = loss_rows.sum(axis=1)
        #     # Create a total column for 'Loss' rows
        #     pivoted_df.loc[loss_rows.index, 'Total'] = total_loss.values


        # Add 'Total' column by summing across the specified columns, only for 'Loss'
        loss_rows = pivoted_df.xs('Loss', level='Type')
        total_loss = loss_rows.sum(axis=1)
        # Create a 'Total' column with the same multi-level index as pivoted_df
        total_column = pd.Series([float('nan')] * len(pivoted_df.index), index=pivoted_df.index)
        total_column.loc[('Loss', slice(None))] = total_loss.values.astype(float)
        pivoted_df['total'] = total_column

        # Sort columns based on the order in self.discriminators
        discriminator_order = ['total'] + ['latent'] + list(self.discriminators.keys()) if 'latent' in pivoted_df.columns else ['total'] + list(self.discriminators.keys())
        pivoted_df = pivoted_df.reindex(columns=discriminator_order)

        # Compute maximum column header length
        num_columns = len(pivoted_df.columns) # +1 for 'Batch'
        space_for_columns = max_row_length - len('Type B  ')
        max_col_length = max(7, space_for_columns // num_columns)

        # Abbreviate discriminator names based on computed length
        pivoted_df.columns = [col[:max_col_length].strip() for col in pivoted_df.columns]

        # Round the data to desired decimal places and fillna
        rounded_df = pivoted_df.round(2).fillna('')

        # Reorder 'Type' values
        type_order = ['Loss', 'Orig', 'Deno']
        rounded_df = rounded_df.reindex(type_order, level='Type')

        # Modify the index to collapse headers
        rounded_df.index = [' '.join(map(str, idx)) for idx in rounded_df.index]

        # Convert DataFrame to string and print
        df_string = rounded_df.to_string(index=True, justify='right')
        tqdm.write(df_string)

    # Inside DiscriminatorManager
    def _process_batch(self, original, denoised, step, timesteps, output_name, output_dir):
        original["decode"], denoised["decode"] = self._decode_latents(original["latent"], denoised["latent"])
        original["preprocess"], denoised["preprocess"] = self._get_processed_images(original["decode"], denoised["decode"])
        original["embedding"], denoised["embedding"] = self._get_image_embeddings(original["preprocess"], denoised["preprocess"])
        
        self._save_image_pairs(original["decode"], denoised["decode"], step, timesteps, output_name, output_dir)
        
        original["decode"].cpu()
        denoised["decode"].cpu()

    def _decode_latents(self, 
                        original_latents: torch.Tensor, 
                        denoised_latents: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes latent tensors into image tensors and rescales them.

        Parameters:
        original_latents (torch.Tensor): The original latent tensors.
        denoised_latents (torch.Tensor): The denoised latent tensors.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors representing the decoded and rescaled original and denoised images.
        """
        def decode_latents_single(latents):
            decodes = []
            for i in range(latents.size(0)):
                latent = latents[i].unsqueeze(0).to(self.vae_model.dtype)
                decode = self.vae_model.decode(latent).sample
                # Rescale from [-1, 1] to [0, 1]
                decode = (decode + 1) / 2
                decode = decode.clamp(0, 1)
                decodes.append(decode)

            return torch.cat(decodes, dim=0)

        return decode_latents_single(original_latents), decode_latents_single(denoised_latents)
    
    def _transform(self, n_px):
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(n_px),
            # lambda image: image.convert("RGB"),
            # Custom normalization for tensors in the range [0, 255]
            transforms.Lambda(lambda x: x.float() / 255.0),
            # ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) 

    def _get_processed_images(self, original_images: torch.Tensor, denoised_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        def get_embeddings(images):
            if len(images.shape) != 4 or images.shape[1] != 3:
                raise ValueError("Input images must be a 4D tensor with shape (B, 3, H, W)")
            preprocessed_images = preprocess(images)
            return preprocessed_images

        preprocess = self._transform(224) # input size for ViT-L/14
        return get_embeddings(original_images), get_embeddings(denoised_images)


    def _get_image_embeddings(self, original_images: torch.Tensor, denoised_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:           
        def get_embeddings(preprocessed_images):
            embeddings = clip_model.encode_image(preprocessed_images)
            return embeddings / embeddings.norm(dim=-1, keepdim=True)
        clip_model, _ = self.vit_model
        return get_embeddings(original_images), get_embeddings(denoised_images)

    def _save_image_pairs(self, original_images, denoised_images, step, timesteps, output_name, output_dir):

        def _save_image_pair(index, original_tensor, denoised_tensor, output_name, save_dir, step, timesteps):
            original_pil = to_pil_image(original_tensor)
            denoised_pil = to_pil_image(denoised_tensor)
            combined_width = original_pil.width + denoised_pil.width
            combined_height = max(original_pil.height, denoised_pil.height)
            combined_image = Image.new('RGB', (combined_width, combined_height))
            combined_image.paste(original_pil, (0, 0))
            combined_image.paste(denoised_pil, (original_pil.width, 0))

            filename = f"{output_name}_{step}_b{index}_ts{timesteps[index].item()}.jpg"
            combined_image.save(os.path.join(save_dir, filename))

        from threading import Thread
        if step % self.save_image_steps != 0:
            return

        save_dir = os.path.join(output_dir, "sample_discriminator")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for i in range(len(original_images)):
            thread = Thread(target=_save_image_pair, args=(i, original_images[i], denoised_images[i], output_name, save_dir, step, timesteps))
            thread.daemon = True  # Set thread as daemon
            thread.start()


# instead of passin gloss func direction  you can pass a lambda to it
# to call it with optional arguments
# so if you set negative_loss_scale to 0
# you dont reward it for doing better than the original image
# if you set it to -1
# you punish any deviation from original score
# positive or negative
# if you set it to 0.5 (default)
# then improvements over original image get negative loss of half the MSE

# discriminator_manager.add_function(
#     "pixel",
#     5,
#     lambda x, y: positive_classifier_mse(x, y, negative_loss_scale=-1),
#     lambda x: x,
#     #"image"
#     "decode"
# )
# discriminator_manager.add_function(
#     "entropy",
#     2.5,
#     positive_classifier_mse,
#     NoiseAnalysis.entropy,
#     #lambda x: NoiseAnalysis.entropy(x) - NoiseAnalysis.kurtosis(x),
#     "noise"
# )
# discriminator_manager.add_function(
#     "ent-kur",
#     1,
#     positive_classifier_mse,
#     #NoiseAnalysis.entropy,
#     lambda x: NoiseAnalysis.entropy(x) - NoiseAnalysis.kurtosis(x),
#     "noise"
# )
# discriminator_manager.add_function(
#     "kurtosis",
#     2.5,
#     lambda x,y: positive_classifier_mse(x, y, negative_loss_scale=-1),
#     NoiseAnalysis.kurtosis,
#     "noise"
# )
# discriminator_manager.add_function(
#     "skewness",
#     2.5,
#     lambda x,y: positive_classifier_mse(x, y, negative_loss_scale=-1),
#     NoiseAnalysis.skewness,
#     "noise"
# )
# discriminator_manager.add_function(
#     "sharpness",
#     0.5,
#     positive_classifier_mse,
#     NoiseAnalysis.sharpness_tensor,
#     #"image"
#     "decode"
# )
# discriminator_manager.add_function(
#     "noise",
#     1,
#     positive_classifier_mse,
#     NoiseAnalysis.noise,
#     "image"
# )
