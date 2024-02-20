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

from prodigyopt import Prodigy
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

timesteps = 1000

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
import clip
import torch.nn as nn
import pytorch_lightning as pl

from library import sdxl_model_util
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import torchvision
from typing import Tuple, List
import numpy as np
import pandas as pd

from diffusers import AutoencoderTiny
import torch.nn.functional as F
import inspect
import itertools



def positive_classifier_mse(original_scores, denoised_scores, power=2, min_original_score=-100, negative_loss_scale=0.5):
    # Check if the tensors have more than one dimension and flatten if needed
    if original_scores.ndim > 1:
        original_scores = original_scores.reshape(original_scores.size(0), -1)
    if denoised_scores.ndim > 1:
        denoised_scores = denoised_scores.reshape(denoised_scores.size(0), -1)

    # Compute the positive loss term only where the mask is True
    positive_loss = torch.maximum(torch.zeros_like(original_scores), original_scores - denoised_scores) ** power
    negative_loss = torch.maximum(torch.zeros_like(original_scores), denoised_scores - original_scores) ** power
    
    # Combine the loss terms, applying the negative_loss_scale only to the negative loss
    loss = positive_loss - negative_loss_scale * negative_loss

    # Only apply loss if it met a minimum criteria
    loss = torch.where(original_scores > min_original_score, loss, torch.zeros_like(loss))

    # Take the mean (or sum) of the loss across all dimensions except the batch dimension
    if original_scores.ndim > 1:
        #loss = loss.mean(dim=1, keepdim=True)
        loss = loss.mean(dim=1)

    return loss


# For BAD scores we simply swap the order
def negative_classifier_mse(original_scores, denoised_scores, power=2, min_original_score=-100, negative_loss_scale=0.5):
    return positive_classifier_mse(denoised_scores, original_scores, power, min_original_score, negative_loss_scale)    


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

    # def _check_embedding_dimension(self, embed_tensor):
    #     expected_emb_dim = text_encoders[0].get_input_embeddings().weight.shape[-1]
    #     if expected_emb_dim != embed_tensor.shape[-1]:
    #         raise ValueError(f"Loaded embeddings are of incorrect shape. Expected {expected_emb_dim}, but are {embed_tensor.shape[-1]}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class GANCriticModel(nn.Module):
    def __init__(self, model_path=None, optimizer_cls=torch.optim.Adam, optimizer_args={"lr": 0.0002}, num_channels=4, output_size=(4, 4)):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            ResidualBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            ResidualBlock(256),  # Added final Residual Block here
            nn.AdaptiveAvgPool2d(output_size)
        )
        flattened_size = 256 * output_size[0] * output_size[1]
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size + 1, 1024),
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
            nn.Sigmoid(),
        )
        self.optimizer = optimizer_cls(self.parameters(), **optimizer_args)
        self.model_path = model_path

        if model_path and os.path.isfile(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"GAN Model loaded from {model_path}")

    def forward(self, x, timesteps):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x, timesteps.unsqueeze(1)), dim=1) 
        return self.linear_layers(x)

    def save_model(self, step_name):
        """Save the model state to the specified file."""
        save_path, ext = os.path.splitext(self.model_path)
        save_path += f"_{step_name}"
        model_path = save_path + ext
        torch.save(self.state_dict(), model_path)
        print(f"GAN Model saved to {model_path}")




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

class BaseDiscriminator:
    def __init__(self, manager, name, weight, loss_func, input_type='embedding'):
        self.manager = manager
        self.name = name
        self.weight = weight
        self.loss_func = loss_func
        self.input_type = input_type
        self.device = manager.device
        self.enable_grad_original_score=False
        self.enable_grad_denoised_score=True
        #manager.add_discriminator(name, self)

    def before_scores(self):
        pass

    def compute_loss(self, original_scores, denoised_scores, original, denoised):
        return self.loss_func(original_scores, denoised_scores)

    def save(self, step_name):
        pass

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



from collections import deque
class GANDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, critic, input_type='noise', max_past_batches=6):
        super().__init__(manager, name, weight, loss_func, input_type=input_type)
        self.manager = manager
        self.critic = manager.accelerator.prepare(critic).to(self.device)
        #self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0002)
        #self.criterion = torch.nn.BCELoss(reduction='none')
        self.enable_grad_original_score=True
        self.past_labels = deque([], maxlen=max_past_batches)  # Hold the n most recent batches

    def before_scores(self):
        tqdm.write("before_scores")
        #with torch.enable_grad():
        self.critic.optimizer.zero_grad()
        self.critic.train()

    def compute_scores(self, latent):
        score = self.critic(latent.detach(), self.manager.timesteps).squeeze(1)
        #print("score", score)
        return score

    def compute_loss(self, real_scores, fake_scores, original, denoised):
        #print("Run GAN")
        # Setup Labels
        b_size = real_scores.shape[0]
        real_labels =  torch.full((b_size,), 0.9, device=self.device)
        fake_labels =  torch.full((b_size,), 0.1, device=self.device)
        # Train the GAN Critic
        real_loss = self.loss_func(real_scores, real_labels).squeeze()
        fake_loss = self.loss_func(fake_scores, fake_labels).squeeze()
        d_loss = (real_loss + fake_loss)
        self.manager.accelerator.backward(d_loss.mean())
        self.critic.optimizer.step()
        self.critic.optimizer.zero_grad()

        for reals, fakes in self.past_labels:
            real_labels_prior =  torch.full((reals[0].shape[0],), 0.9, device=self.device)
            fake_labels_prior =  torch.full((reals[0].shape[0],), 0.1, device=self.device)
            real_scores = self.critic(reals[0].detach().to(self.device), reals[1]).squeeze(1)
            fake_scores = self.critic(fakes[0].detach().to(self.device), fakes[1]).squeeze(1)
            real_loss = self.loss_func(real_scores, real_labels_prior).squeeze()
            fake_loss = self.loss_func(fake_scores, fake_labels_prior).squeeze()
            d_loss = real_loss + fake_loss
            self.manager.accelerator.backward(d_loss.mean())
            self.critic.optimizer.step()
            self.critic.optimizer.zero_grad()
            reals[0]=reals[0].cpu()
            fakes[0]=fakes[0].cpu()

        self.past_labels.append(([original["latent"], self.manager], [denoised["latent"], self.manager]))

        # Combine losses and update
        # print(f"GAN Critic Scores: Real/Fake:", real_scores, fake_scores)
        # print(f"GAN Critic Losses: Real/Fake:", real_loss, fake_loss)
        # print(f"GAN Critic Loss:", d_loss)
        #accelerator.backward(d_loss.mean())  # Using accelerator for gradients computation
        #self.critic.optimizer.step()

        self.critic.eval()

        #Generate Critic Loss for Generator
        gen_fake_scores = self.critic(denoised["latent"], self.manager).squeeze(1)
        gen_fake_loss = self.loss_func(gen_fake_scores, real_labels).squeeze()
        print(f"GAN Gen Scores: Fake:", gen_fake_scores)
        print(f"GAN Critic Loss:", gen_fake_loss)
        #gen_fake_loss = torch.clamp(gen_fake_loss, min=-float('inf'), max=0.2)
        return gen_fake_loss

    def save(self, global_step):
        self.critic.save_model(global_step)




class AestheticDiscriminator(BaseDiscriminator):
    def __init__(self, manager, name, weight, loss_func, model):
        super().__init__(manager, name, weight, loss_func)
        self.model = model.to(self.device)

    def compute_scores(self, embeddings):
        return self.model(embeddings).squeeze() / 10

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

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
           

class DiscriminatorManager:
    # gradient_types = ["noise", "latent", "decode", "embedding"]
    gradient_types = ["noise", "latent"]  #, "decode"
    timesteps = 1000
    step = 0
    def __init__(self, config_json, device, noise_scheduler, tokenizers, text_encoders, is_sdxl, save_image_steps=10, print_diagnostics=True, accelerate=None, step=None, args={}):
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
        self.accelerate = accelerate
        self.args = args
        
        if is_sdxl:
            sdxl_model_util.VAE_SCALE_FACTOR = 1.0
            self.vae_scale_factor = 1.0
            print('VAE = SDXL')
            self.vae_model = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to(device)
        else:
            print('VAE = SD15')
            self.vae_model = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)
        
        clip_model, _ = self.vit_model

        if "decode" in self.gradient_types or "embedding" in self.gradient_types:
            self.vae_model.requires_grad_(True)
            for param in self.vae_model.parameters():
                param.requires_grad = True

            if "embedding" in self.gradient_types:
                clip_model.requires_grad_(True)
                clip_model.train()

                for param in clip_model.parameters():
                    param.requires_grad = True
        
        self.add_aesthetics(self.config_json["aesthetics"])
        self.add_blipscores(self.config_json["blipScores"])
        self.add_clipscores(self.config_json["clipScores"])
        self.add_image_rewards(self.config_json["imageRewards"])
        self.add_functions(self.config_json["functions"])
        self.add_embeddings(self.config_json["embeddings"])
        self.add_embedding_texts(self.config_json["embeddingTexts"])

    def todevice(self):
        clip_model, _ = self.vit_model
        clip_model.to(self.device)
        self.vae_model.to(self.device)
        for discriminator_name, discriminator in self.discriminators.items():  # Make sure to call .items()
            if hasattr(discriminator, 'model'):
                discriminator.model.to(self.device)

    def save(self, step_name):
        for discriminator_name, discriminator in self.discriminators.items():  # Make sure to call .items()
            discriminator.save(step_name)      

    def classifier_lambda(self, obj):
        
        if obj["classifierType"] == "predefined":
            if "NoiseAnalysis.pyramid_loss" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[1, 1], steps=-1, kernel_type="avg_ind")
            if "kl_mv" in obj["classifierExpression"]:
                lmbda = lambda x, y: multivariate_kl_divergence_grid(x, y)
            if "antiblur" in obj["classifierExpression"]:
                lmbda = lambda x, y: - F.mse_loss(NoiseAnalysis.blur_latent(y), y, reduction="none").mean([1, 2, 3])
            if "kl_mvl" in obj["classifierExpression"]:
                lmbda = lambda x, y: multivariate_kl_divergence_grid(x, y, 11, 0.8)
            if "kl_fft" in obj["classifierExpression"]:
                lmbda = lambda x, y: -multivariate_kl_divergence_grid(x, y, 12, 1, "FFT")
            if "batch_var" in obj["classifierExpression"]:
                lmbda = lambda label, pred: batch_var(label, pred)
            if "batch_covar" in obj["classifierExpression"]:
                lmbda = lambda label, pred: batch_covar(label, pred)
            if "batch_var_fft" in obj["classifierExpression"]:
                lmbda = lambda label, pred: batch_var_fft(label, pred)
            if "batchdiff" in obj["classifierExpression"]:
                lmbda = lambda label, pred: (0.999 * pairwise_loss(pred, label) - pairwise_loss(pred)).mean(dim=[1, 2])
            if "reldiff" in obj["classifierExpression"]:
                lmbda = lambda label, pred: (pairwise_loss(label).log() - pairwise_loss(pred).log()).mean(dim=[1, 2])
            if "kl_all" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.kl_divergence(x.float(), y.float(), kernel_size=None, kernel_type="avg_ind")
            if "kl1_fft" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.kl_divergence(x.float(), y.float(), kernel_size=1, kernel_type="avg_ind")
            if "kl_fft" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.kl_divergence(x.float(), y.float(), kernel_size=None, kernel_type="avg_ind")
            if "kl3c" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[3, 3], steps=-1, kernel_type="avg_comb")
            if "kl3g" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[3, 3], steps=-1, kernel_type="gaussian")
            if "kl3gb" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[3, 3], steps=-1, kernel_type="gabor")
            if "kl4" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[4, 4], steps=-1)
            if "pixel" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), max_level=1, min_kernel_size=1, max_kernel_size=7, eps=0.1, dilations=False, func=NoiseAnalysis.kl_divergence)
            if "sat_pixel" in obj["classifierExpression"]:
                lmbda = lambda x, y: positive_classifier_mse(x, y, negative_loss_scale=0.1)
            if "contrast" in obj["classifierExpression"]:
                lmbda = lambda x, y: positive_classifier_mse(x, y, negative_loss_scale=0.1)
            if "kl_fft" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x, y, max_level=1, min_kernel_size=2, max_kernel_size=7, dilations=True, func=NoiseAnalysis.kl_divergence)
            if "kl_n2" in obj["classifierExpression"]:
                lmbda = NoiseAnalysis.kl_divergence
            if "kl_fft_alt" in obj["classifierExpression"]:
                lmbda = lambda x, y: 1e-4 * F.mse_loss(x.float(), y.float(), reduction="none").mean([1, 2, 3])
            if "con_g" in obj["classifierExpression"]:
                lmbda = lambda x, y: F.mse_loss(x.float(), y.float(), reduction="none")
            if "kl1_l" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x.float(), y.float(), func=NoiseAnalysis.kl_divergence, kernel_range=[1, 1], steps=-1, cw=[1,1,1,1],kernel_type="avg_ind")
            if "sat_p" in obj["classifierExpression"]:
                lmbda = lambda x, y: F.mse_loss(x.float(), y.float(), reduction="none").mean([1,2,3])
            if "con_p" in obj["classifierExpression"]:
                lmbda = lambda x, y: F.mse_loss(x.float(), y.float(), reduction="none").mean([1,2,3])
            if "sat_n" in obj["classifierExpression"]:
                lmbda = lambda x, y: F.mse_loss(x.float(), y.float(), reduction="none")
            if "kl_gabor_l" in obj["classifierExpression"]:
                lmbda = NoiseAnalysis.kl_divergence
            if "vit" in obj["classifierExpression"]:
                lmbda = lambda x, y: (1-F.cosine_similarity(x, y, dim=-1))**2
            if "edges" in obj["classifierExpression"]:
                lmbda = lambda x, y: F.mse_loss(x, y, reduction='none').mean([1, 2, 3])
            if "fft_kl_d" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x, y, max_level=1, min_kernel_size=5, max_kernel_size=25, dilations=True, func=NoiseAnalysis.kl_divergence, eps=1e-3, equal_weighting=True)
            if "fft_klc" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x, y, max_level=1, min_kernel_size=5, max_kernel_size=25, dilations=True, func=NoiseAnalysis.kl_divergence_comb, eps=1e-3, equal_weighting=True)
            if "ssim" in obj["classifierExpression"]:
                lmbda = lambda x, y: NoiseAnalysis.pyramid_loss(x, y, max_level=1, min_kernel_size=2, max_kernel_size=15, func=NoiseAnalysis.ssim_divergence)
            if "BCELoss" in obj["classifierExpression"]:
                lmbda = torch.nn.BCELoss(reduction='none')
            if "gan" in obj["classifierExpression"]:
                lmbda = torch.nn.BCELoss(reduction='none')



                
    #NoiseAnalysis.kl_divergence
        elif obj["classifierType"] == "custom":
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
            if "passthrough_float" in function["functionType"]:
                function_lambda = lambda x: x.float()
            if "passthrough" in function["functionType"]:
                function_lambda = lambda x: x
            elif "fft_stack"  in function["functionType"]:
                function_lambda = lambda x: fft_stack(x)
            elif "sat_pixel" in function["functionType"]:
                function_lambda = lambda x: NoiseAnalysis.saturation(x, reduce_to_mean=False)
            elif "contrast" in function["functionType"]:
                function_lambda = NoiseAnalysis.contrast
            elif "fft_stack_transpose_stack" in function["functionType"]:
                function_lambda = lambda x: fft_stack(transpose_stack(x))
            elif "con_g" in function["functionType"]:
                function_lambda = lambda x: 4*latents_to_rgb(x).std(dim=[2,3], keepdim=True).mean([1,2,3])
            elif "kl1_l" in function["functionType"]:
                function_lambda = lambda x: latents_to_rgb(x)
            elif "sat_p" in function["functionType"]:
                function_lambda = lambda x: 8*latents_to_rgb(x).std(dim=1, keepdim=True)
            elif "sat_p" in function["functionType"]:
                function_lambda = lambda x: 4*latents_to_rgb(x).std(dim=[2,3], keepdim=True)
            elif "sat_g" in function["functionType"]:
                function_lambda = lambda x: 8*latents_to_rgb(x).std(dim=1, keepdim=True).mean([1,2,3])
            elif "sat_2" in function["functionType"]:
                function_lambda = lambda x: (1/64)*latents_to_rgb(x, latent_scale=2).std(dim=1, keepdim=True).mean([1,2,3])
            elif "sat_3" in function["functionType"]:
                function_lambda = lambda x: (1/64)*latents_to_rgb(x, latent_scale=3).std(dim=1, keepdim=True).mean([1,2,3])
            elif "sat_4" in function["functionType"]:
                function_lambda = lambda x: (1/64)*latents_to_rgb(x, latent_scale=4).std(dim=1, keepdim=True).mean([1,2,3])
            elif "con_n" in function["functionType"]:
                function_lambda = lambda x: 4*latents_to_rgb(x).mean(dim=1, keepdim=True).mean([1,2,3])
            elif "kl_gabor_l" in function["functionType"]:
                function_lambda = lambda x: tensor_func_stack(
                    x.float(), 
                    gabor_filter,
                    kwargs_list = [
                        {"frequency": freq, "sigma": sigma, "theta": theta, "phase": phase}
                        for freq, sigma, theta, phase in itertools.product(
                            [0.01, 0.05, 0.1, 0.2], 
                            [1, 5], 
                            [0, (1/4)*np.pi, (1/2)*np.pi, (3/4)*np.pi],
                            [0],
                        )
                    ]
                )
            elif "edges" in function["functionType"]:
                function_lambda = sobel_edge_grayscale
            # elif "fft_to_amplitude_phase" in function["functionType"]:
            #     function_lambda = lambda x: amplitude(torch.fft.fftshift(torch.fft.rfft2(x, dim=(2, 3)), dim=(2, 3)))
            elif "gan" in function["functionType"]:
                function_lambda = GANCriticModel(
                    #model_path=r"G:\Stable Diffusion\Training\Classification\gan_diffusion_critic\diffusion_critic_1500.pt",
                    model_path=r"G:\Stable Diffusion\Training\Classification\gan_diffusion_critic\diffusion_critic_ts_v3_400_750_5100_3000.pt",
                    # optimizer_cls=torch.optim.Adam, 
                    # optimizer_args={"lr": 0.0002}, 
                    optimizer_cls=Prodigy, 
                    optimizer_args={
                        "lr":1, 
                        "safeguard_warmup":True, 
                        "use_bias_correction":True,
                        "d_coef":2.0,
                        "weight_decay":0.45,
                        #d_coef=0.1,
                        #weight_decay=0.01,
                        #betas=(0.9, 0.99),
                        "decouple":True,
                    }, 
                )
                 
                
                
                
                
                
            elif "NoiseAnalysis" in function["functionType"]:
                if "sharpness_tensor" in function["functionType"]:
                    function_lambda = NoiseAnalysis.sharpness_tensor
                elif "entropy" in function["functionType"]:
                    function_lambda = NoiseAnalysis.entropy
                elif "kurtosis" in function["functionType"]:
                    function_lambda = NoiseAnalysis.kurtosis
                elif "variance" in function["functionType"]:
                    function_lambda = NoiseAnalysis.variance
                elif "mean" in function["functionType"]:
                    function_lambda = NoiseAnalysis.mean
            if "clipped" in function["functionType"]:
                function_lambda = lambda x: NoiseAnalysis.clipped_latents(x)        
            
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
    
    def add_gan(self, name, weight, loss_func, model, input_type="latent"):
        self.discriminators[name] = GANDiscriminator(self, name, weight, loss_func, model, input_type)
    
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

    @classmethod
    def update_timesteps(cls, curtimesteps):
        global timesteps
        cls.timesteps = curtimesteps
        timesteps = curtimesteps
        
        
    @classmethod
    def update_current_step(cls, step):
        cls.step = step

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

    def scale_losses(self, loss, input_type, timesteps):
        # Since we've converted from noise back to image space, we should use inverse of debiased estimation 
        # to obtain a uniform loss scale across timesteps
        with torch.no_grad():
            if input_type == 'noise':
                if self.args.min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, self.args.min_snr_gamma, self.args.v_parameterization)
                if self.args.scale_v_pred_loss_like_noise_pred:
                    loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, self.noise_scheduler)
                if self.args.v_pred_like_loss:
                    loss = add_v_prediction_like_loss(loss, timesteps, self.noise_scheduler, self.args.v_pred_like_loss)
                if self.args.debiased_estimation_loss:
                    loss = apply_debiased_estimation(loss, timesteps, self.noise_scheduler)
            # else:
            #     ones = torch.ones_like(timesteps)
            #     scaling_factors = apply_debiased_estimation(ones, timesteps, self.noise_scheduler)
            #     scaling_factors = 1.0 / (scaling_factors + 1e-7)
            #     k = 0.05
            #     scaling_factors *= k * ones + (1 - k) * torch.abs(timesteps - 500) / 500 

            #     loss *= scaling_factors


        # Step 4: Multiply the original losses by the inverse scaling factor
        return loss

    def calc_needed_resources(self):
        # Optimize and save memory if we don't need all steps
        self.needs_latent, self.needs_decode, self.needs_embedding = False, False, False
        for discriminator_name, d in self.discriminators.items():
            if d.input_type in ["latent","decode", "embedding"]:
                self.needs_latent = True
                if d.input_type in ["decode", "embedding"]:
                    self.needs_decode = True
                    if d.input_type == "embedding":
                        self.needs_embedding = True
        if not self.needs_decode and self.vae_model:
            self.vae_model.to("cpu") # move it to cpu because if we're generating image samples we still use it
            self.vae_model = None
        if not self.needs_embedding and self.vit_model:
            self.vit_model = None

    def apply_discriminator_losses(self, base_loss, timesteps, original_latents, noise, noisy_latents, noise_pred, step, output_name, output_dir, captions):
        
        def compute_discriminator_loss(discriminator, discriminator_name, original, denoised, timesteps, base_loss, captions):
            # import contextlib
            # @contextlib.contextmanager
            # def conditional_grad(input_type):
            #     if input_type in self.gradient_types:
            #         with torch.enable_grad():
            #             yield
            #     else:
            #         with torch.no_grad():
            #             yield
            discriminator.before_scores()
            with torch.set_grad_enabled(discriminator.input_type in self.gradient_types and discriminator.enable_grad_original_score):
                if discriminator.enable_grad_original_score:
                    # with conditional_grad(discriminator.input_type):       
                    if 'captions' in signature(discriminator.compute_scores).parameters:
                        original_scores = discriminator.compute_scores(original[discriminator.input_type], captions)
                    else:
                        original_scores = discriminator.compute_scores(original[discriminator.input_type])
                else:
                    if 'captions' in signature(discriminator.compute_scores).parameters:
                        original_scores = discriminator.compute_scores(original[discriminator.input_type], captions)
                    else:
                        original_scores = discriminator.compute_scores(original[discriminator.input_type]).detach()

            #with torch.set_grad_enabled(discriminator.input_type in self.gradient_types or discriminator.enable_grad_denoised_score):   
            with torch.set_grad_enabled(discriminator.input_type in self.gradient_types):
                if 'captions' in signature(discriminator.compute_scores).parameters:
                    denoised_scores = discriminator.compute_scores(denoised[discriminator.input_type], captions)
                else:
                    denoised_scores = discriminator.compute_scores(denoised[discriminator.input_type])
                        
                discriminator_loss = discriminator.compute_loss(original_scores, denoised_scores, original, denoised)
                discriminator_loss = self.scale_losses(discriminator_loss, discriminator.input_type, timesteps)
                discriminator_loss *= discriminator.weight
                
            diagnostics = self._accumulate_diagnostics(discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss)
            return discriminator_loss, discriminator, diagnostics

        #with torch.no_grad():
        with torch.set_grad_enabled(True):
            #self.todevice()
            original = {}
            denoised = {}
            #self.denoised = denoised

            original["noise"] = noise.float()
            denoised["noise"] = noise_pred.float()

            self.calc_needed_resources()

            if self.needs_latent:
                original["latent"] = original_latents
                with torch.no_grad():
                    original["latent"] = self.remove_noise(noisy_latents, noise, timesteps)
                with torch.set_grad_enabled("latent" in self.gradient_types):
                    denoised["latent"] = self.remove_noise(noisy_latents, noise_pred, timesteps)
                
                # Decode latents and get images, embeddings
                self._process_batch(original, denoised)

            all_diagnostics = []
            gradient_loss = torch.zeros_like(base_loss)
            #gradient_loss.requires_grad_()
            ungradient_loss = torch.zeros_like(base_loss)
            for discriminator_name, discriminator in self.discriminators.items():
                print(discriminator_name)
                discriminator_loss, discriminator, diagnostics = compute_discriminator_loss(discriminator, discriminator_name, original, denoised, timesteps, base_loss, captions)
                #print(discriminator_name, discriminator_loss)
                if discriminator_loss.ndim == 0:
                    discriminator_loss = discriminator_loss.unsqueeze(0)
                if discriminator.input_type in self.gradient_types:
                    gradient_loss += discriminator_loss
                #else:
                ungradient_loss += discriminator_loss 
                all_diagnostics.extend(diagnostics)

            # Pivoting and displaying the data
            if self.print_diagnostics:
                df = pd.DataFrame(all_diagnostics)
                self._pivot_and_display(df)

            # Compute the scaling factor
            epsilon = 1e-7
            #modified_loss_scale = (base_loss + gradient_loss + ungradient_loss/10 * (timesteps * timesteps / 1e6 ) + epsilon) / (base_loss + gradient_loss + epsilon)
            modified_loss_scale = 1
            # for k in original:
            #     original[k].cpu()
            #     denoised[k].cpu()
            #self.tocpu()

        # gradient_loss.mean().backward(retain_graph=True)
        # # Print gradients
        # print("Gradient of noise:", denoised["noise"].grad)
        # print("Gradient of latent:", denoised["latent"].grad)
        # print("Gradient of decode:", denoised["decode"].grad)
        # print("Gradient of embedding:", denoised["embedding"].grad)

        #final_loss = (base_loss + gradient_loss) * modified_loss_scale.detach()
        #base_loss = 0
        final_loss = base_loss + gradient_loss + ungradient_loss.detach()
        #final_loss = (base_loss + gradient_loss) * modified_loss_scale + ungradient_loss.detach()
        #final_loss = gradient_loss
        #final_loss = base_loss + gradient_loss + ungradient_loss.detach()

        # denoised["noise"].retain_grad()
        # denoised["latent"].retain_grad()
        # denoised["decode"].retain_grad()
        # denoised["embedding"].retain_grad()

        
        return final_loss

    def _accumulate_diagnostics(self, discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss):
        with torch.no_grad():
            diagnostic_data = []

            # Ensure tensors are 2D (batch_size x data)
            original_scores = original_scores.reshape(-1, 1)
            denoised_scores = denoised_scores.reshape(-1, 1)
            discriminator_loss = discriminator_loss.view(-1, 1)
            base_loss = base_loss.view(-1, 1)
            timesteps = timesteps.view(-1, 1)

            # Convert to numpy arrays for easier processing
            original_scores_np = original_scores.cpu().numpy()
            denoised_scores_np = denoised_scores.cpu().numpy()
            discriminator_loss_np = discriminator_loss.cpu().numpy()
            base_loss_np = base_loss.cpu().numpy()
            timesteps_np = timesteps.cpu().numpy()
            
            print('discriminator_loss_np.shape[0]')
            print(discriminator_loss_np.shape[0])
            print(timesteps_np.shape[0])
            print(discriminator_loss)
            print(timesteps)

            for i in range(discriminator_loss_np.shape[0]):
                # Check if the scores are simple floats (ndim == 2 and size == 1 for 2nd dim)
                #if original_scores.ndim == 2 and original_scores.size(1) == 1 and denoised_scores.ndim == 2 and denoised_scores.size(1) == 1:
                if discriminator_loss_np.shape[0] == original_scores_np.shape[0]:
                    diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Orig", "Value": original_scores_np[i, 0], "TS": timesteps_np[i, 0]})
                    diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Deno", "Value": denoised_scores_np[i, 0], "TS": timesteps_np[i, 0]})

                # Always add the loss
                v = timesteps_np[i, 0]
                d = discriminator_loss_np[i, 0]
                diagnostic_data.append({"Batch": i, "Discriminator": discriminator_name, "Type": "Loss", "Value": d, "TS": v})

                # Add original latent loss (only once per batch, not per discriminator)
                diagnostic_data.append({"Batch": i, "Discriminator": "mse", "Type": "Loss", "Value": base_loss_np[i, 0], "TS": timesteps_np[i, 0]})

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
        discriminator_order = ['total'] + ['mse'] + list(self.discriminators.keys()) if 'mse' in pivoted_df.columns else ['total'] + list(self.discriminators.keys())
        pivoted_df = pivoted_df.reindex(columns=discriminator_order)

        # Compute maximum column header length
        num_columns = len(pivoted_df.columns) # +1 for 'Batch'
        space_for_columns = max_row_length - len('Type B  ')
        max_col_length = max(7, space_for_columns // num_columns)

        # Abbreviate discriminator names based on computed length
        pivoted_df.columns = [col[:max_col_length].strip() for col in pivoted_df.columns]

        # Round the data to desired decimal places and fillna
        rounded_df = pivoted_df.round(3).fillna('')

        # Reorder 'Type' values
        type_order = ['Loss', 'Orig', 'Deno']
        rounded_df = rounded_df.reindex(type_order, level='Type')

        # Modify the index to collapse headers
        rounded_df.index = [' '.join(map(str, idx)) for idx in rounded_df.index]

        # Convert DataFrame to string and print
        #pivoted_df = pivoted_df.style.format({key: '{:.3f}' for key in pivoted_df.columns})
        df_string = rounded_df.to_string(index=True, justify='right', float_format="{:.3f}".format)
        tqdm.write(df_string)

    # Inside DiscriminatorManager
    def _process_batch(self, original, denoised):

        if self.needs_decode:
            original["decode"], denoised["decode"] = self._decode_latents(original["latent"], denoised["latent"])
            original["preprocess"], denoised["preprocess"] = self._get_processed_images(original["decode"], denoised["decode"])

            #print("original", original["decode"])
            #print("denoised", denoised["decode"])
            self._save_image_pairs(original["decode"], denoised["decode"])
            if self.needs_embedding:
                original["embedding"], denoised["embedding"] = self._get_image_embeddings(original["preprocess"], denoised["preprocess"])
            #original["decode"].cpu()
            #denoised["decode"].cpu()

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
                #print("vae_model.dtype", self.vae_model.dtype)
                decode = self.vae_model.decode(latent).sample #.to(latents.dtype)
                # Rescale from [-1, 1] to [0, 1]
                decode = (decode + 1) / 2
                decode = decode.clamp(0, 1)
                decodes.append(decode)

            return torch.cat(decodes, dim=0)

        with torch.no_grad():
            original_decode = decode_latents_single(original_latents).detach()

        with torch.set_grad_enabled("decode" in self.gradient_types):
            denoised_decode = decode_latents_single(denoised_latents)

        return original_decode, denoised_decode

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

        def resize(images, size=224, mode='bicubic'):
            # Assuming images are in shape (batch_size, channels, height, width)
            # `mode` must be one of: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'
            #resized_images = F.interpolate(images, size=(size, size), mode=mode, align_corners=False if mode != 'nearest' else None)
            resized_images = F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)
            return resized_images
            # Example using bilinear interpolation

        def center_crop(images, new_height=224, new_width=224):
            height, width = images.shape[2], images.shape[3]
            startx = width // 2 - (new_width // 2)
            starty = height // 2 - (new_height // 2)    
            return images[:, :, starty:starty+new_height, startx:startx+new_width]

        def normalize(images, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
            # Assuming mean and std are tuples of 3 elements for 3 channels
            mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
            std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
            return (images - mean) / std

        def get_embeddings(images):
            #preprocessed_images = self.clip_preprocess(images)
            preprocessed_images = normalize(center_crop(resize(images)))
            embeddings = clip_model.encode_image(preprocessed_images)
            return embeddings / embeddings.norm(dim=-1, keepdim=True)

        clip_model, _ = self.vit_model
        
        with torch.no_grad():
            original_embeddings = get_embeddings(original_images).detach()
        #with torch.set_grad_enabled("embedding" in self.gradient_types):

        with torch.set_grad_enabled("embedding" in self.gradient_types):
            clip_model.requires_grad_("embedding" in self.gradient_types)
            denoised_embeddings = get_embeddings(denoised_images)

        return original_embeddings, denoised_embeddings

    def _save_image_pairs(self, original_images, denoised_images):

        def _save_image_pair(index, original_tensor, denoised_tensor, save_dir, step, timesteps):
            original_pil = to_pil_image(original_tensor)
            denoised_pil = to_pil_image(denoised_tensor)
            combined_width = original_pil.width + denoised_pil.width
            combined_height = max(original_pil.height, denoised_pil.height)
            combined_image = Image.new('RGB', (combined_width, combined_height))
            combined_image.paste(original_pil, (0, 0))
            combined_image.paste(denoised_pil, (original_pil.width, 0))

            filename = f"{self.args.output_name}_{step}_b{index}_ts{timesteps[index].item()}.jpg"
            combined_image.save(os.path.join(save_dir, filename))

        from threading import Thread
        print('self.step % self.save_image_steps != 0')
        print(self.step % self.save_image_steps != 0)
        if self.step % self.save_image_steps != 0:
            return

        save_dir = os.path.join(self.args.output_dir, "sample_discriminator")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for i in range(len(original_images)):
            thread = Thread(target=_save_image_pair, args=(i, original_images[i], denoised_images[i], save_dir, self.step, self.timesteps))
            thread.daemon = True  # Set thread as daemon
            thread.start()


class NoiseAnalysis:

    @classmethod
    def kurtosis(cls, tensor):
        # Ensure tensor is in float format for mean and variance computations
        tensor = tensor.float()

        # Flatten the tensor except for the batch dimension
        flattened = tensor.view(tensor.shape[0], -1)
        
        # Calculate the mean and standard deviation along the flattened dimension
        mean = flattened.mean(dim=1, keepdim=True)
        std = flattened.std(dim=1, unbiased=True, keepdim=True)
        
        # Avoid division by zero by creating a new tensor for std, replacing 0 with 1
        #std_replaced = torch.where(std == 0, torch.ones_like(std), std)

        # Compute the fourth moment (the numerator of the kurtosis formula)
        fourth_moment = ((flattened - mean) ** 4).mean(dim=1)
        
        # Compute the kurtosis for each item in the batch using the adjusted std tensor
        kurt = fourth_moment / (std.squeeze(1) ** 4)
        
        # Adjust for excess kurtosis
        excess_kurtosis = kurt - 3
        
        return excess_kurtosis

    @classmethod
    def entropy(cls, tensor):
        # Reshape tensor to [batch, -1]
        flattened = tensor.view(tensor.shape[0], -1)

        # Compute standard deviation along the last dimension
        std_dev = torch.std(flattened, dim=1)

        # Calculate differential entropy for a normal distribution
        entropies = 0.5 * torch.log2(2 * np.pi * np.e * std_dev ** 2)

        return entropies

    @classmethod
    def mean(cls, tensor):
        # Flatten tensor to [batch, -1]
        flattened = tensor.view(tensor.shape[0], -1)
        # Compute mean along the last dimension
        means = torch.mean(flattened, dim=1)
        return means

    @classmethod
    def stddev(cls, tensor):
        std_devs = torch.std(tensor.view(tensor.shape[0], -1), dim=1)
        return std_devs.to(tensor.device)

    @classmethod
    def variance(cls, tensor):
        std_devs = torch.std(tensor.view(tensor.shape[0], -1), dim=1)
        return std_devs.to(tensor.device) ** 2

    @classmethod
    def skewness(cls, tensor):
        # Flatten tensor to [batch, -1]
        flattened = tensor.view(tensor.shape[0], -1)
        # Compute mean along the last dimension
        mean = torch.mean(flattened, dim=1, keepdim=True)
        # Compute the third moment (central moment)
        third_moment = torch.mean((flattened - mean) ** 3, dim=1)
        # Compute standard deviation along the last dimension
        std_dev = torch.std(flattened, dim=1)
        # Normalize the third moment by the standard deviation cubed
        skewness = third_moment / (std_dev ** 3)
        return skewness

    @classmethod
    def contrast(cls, tensor):
        # Assuming tensor is in BCHW format and RGB color space
        # Convert to grayscale by averaging across channels
        grayscale_approx = torch.mean(tensor, dim=1)
        # Calculate the standard deviation of the grayscale approximation for each image
        contrast_values = torch.std(grayscale_approx.view(tensor.shape[0], -1), dim=1)
        return 4*contrast_values

    @classmethod
    def saturation(cls, tensor, reduce_to_mean=True):
        # Assuming tensor is in BCHW format and RGB color space
        # Calculate the standard deviation across color channels for each pixel, then average those
        std_devs = torch.std(tensor, dim=1)  # Standard deviation across channels
        if reduce_to_mean:
            # Reduce to a single value per batch item
            avg_saturation = 8 * std_devs.mean(dim=[1, 2, 3])  # Average across height and width
            return avg_saturation #.unsqueeze(0)  # Unsqueeze to maintain B x 1 shape for consistency
            #return 8 * torch.mean(std_devs, dim=[1, 2])
        else:
            # Return per-pixel saturation, scaled up by 8 as per your original function
            return 8 * std_devs
        #avg_saturation = torch.mean(std_devs, dim=[1, 2])  # Average across height and width
        #return 8*avg_saturation

    @classmethod
    def blur_latent(cls, latent_tensor):
        def gaussian_blur_kernel():
            # Define a 3x3 Gaussian blur kernel
            kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
            kernel /= kernel.sum()
            return kernel
        # Get the Gaussian blur kernel
        kernel = gaussian_blur_kernel().to(latent_tensor.device)

        # Reshape the kernel to match the shape expected by conv2d: (out_channels, in_channels, height, width)
        # Assuming the latent_tensor has shape [Batch, Channels, Height, Width]
        channels = latent_tensor.shape[1]
        kernel = kernel.expand(channels, 1, 3, 3)

        # Apply padding to maintain the size
        padding = 1  # This is suitable for a 3x3 kernel

        # Apply the blur to each channel
        blurred_tensor = torch.nn.functional.conv2d(latent_tensor, kernel, padding=padding, groups=channels)
        return blurred_tensor

    @classmethod
    def ssim_divergence(cls, tensor1, tensor2, kernel_size=None, stride=1, dilation=1, C1=0.01**2, C2=0.03**2, eps=1e-4):
        avg_kernel, padding = cls.prepare_convolution_params(tensor1, kernel_size)

        n_channels = tensor1.size(1)
        # Compute means using convolutions
        mean1 = F.conv2d(tensor1, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=n_channels)
        mean2 = F.conv2d(tensor2, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=n_channels)

        # Compute square of means and their product
        mean1_sq = mean1.pow(2)
        mean2_sq = mean2.pow(2)
        mean1_mean2 = mean1 * mean2

        # Compute variances and covariance using convolutions
        sigma1_sq = F.conv2d(tensor1 * tensor1, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=n_channels) - mean1_sq
        sigma2_sq = F.conv2d(tensor2 * tensor2, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=n_channels) - mean2_sq
        sigma12 = F.conv2d(tensor1 * tensor2, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=n_channels) - mean1_mean2

        # Ensure positive variance for stability
        sigma1_sq = torch.clamp(sigma1_sq, min=eps)
        sigma2_sq = torch.clamp(sigma2_sq, min=eps)

        # Compute SSIM index
        ssim_numerator = (2 * mean1_mean2 + C1) * (2 * sigma12 + C2)
        ssim_denominator = (mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_numerator / ssim_denominator

        # Stabilize SSIM value before taking -log to avoid -log(0)
        ssim_val = ssim_map.clamp(min=eps)  # Ensure SSIM is not zero
        #neg_log_ssim = -torch.log(ssim_val).mean(dim=[1, 2, 3])  # -log and average over all patches
        neg_log_ssim = ((1-ssim_val)**2).mean(dim=[1, 2, 3])  # -log and average over all patches

        return neg_log_ssim

    @classmethod
    def prepare_convolution_params(cls, tensor, kernel_size=None, mode='independent'):
        channels, height, width = tensor.size(1), tensor.size(2), tensor.size(3)
        if kernel_size is None:
            kernel_size1, kernel_size2 = height, width  # Corrected typo
            padding = 0
        else:
            kernel_size1, kernel_size2 = kernel_size, kernel_size
            padding = kernel_size // 2

        if mode == 'combined':
            # Create a kernel that spans all channels
            avg_kernel = torch.ones((1, channels, kernel_size1, kernel_size2), device=tensor.device) / (channels * kernel_size1 * kernel_size2)
        else:
            # Create individual kernels for each channel
            avg_kernel = torch.ones((channels, 1, kernel_size1, kernel_size2), device=tensor.device) / (kernel_size1 * kernel_size2)

        return avg_kernel, padding

    # @classmethod
    # def local_mmd(tensor2, tensor1, avg_kernel, padding, sstride=1, dilation=1, sigma=2.0):
    #     def gaussian_kernel(a, b, sigma=2.0):
    #         """Compute the Gaussian kernel between a and b"""
    #         sq_dist = (a - b).pow(2).sum(1)
    #         return torch.exp(-0.5 / (sigma ** 2) * sq_dist)

    #     avg_kernel, padding = cls.prepare_convolution_params(tensor1, kernel_size, mode)

    #     groups = tensor1.size(1)

    #     # Compute local embeddings using convolution
    #     local_embed_tensor1 = F.conv2d(tensor1, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=groups)
    #     local_embed_tensor2 = F.conv2d(tensor2, avg_kernel, padding=padding, stride=stride, dilation=dilation, groups=groups)
        
    #     # Flatten embeddings to simplify kernel computation
    #     # Assuming avg_kernel produces a single channel output for simplicity
    #     le_t1_flat = local_embed_tensor1.view(local_embed_tensor1.size(0), -1, 1)
    #     le_t2_flat = local_embed_tensor2.view(local_embed_tensor2.size(0), -1, 1)
        
    #     # Compute Gaussian kernel between all pairs of local embeddings
    #     kernel_vals = gaussian_kernel(le_t1_flat, le_t2_flat, sigma=sigma)
        
    #     # Compute MMD by aggregating kernel evaluations
    #     mmd = kernel_vals.mean(dim=[1, 2, 3])
        
    #     return mmd

    @classmethod
    def make_gaussian_kernel(cls, size: int, sigma: float):
        """Generates a 2D Gaussian kernel."""
        if size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        
        # Create a grid of (x,y) coordinates
        x = torch.arange(size) - (size - 1) / 2
        y = x.view(-1, 1)
        x2y2 = x**2 + y**2

        # Compute the Gaussian kernel
        g = torch.exp(-(x2y2 / (2 * sigma**2)))
        g /= g.sum()
        
        # Make it 4D for convolutional operations: (out_channels, in_channels/groups, height, width)
        return g.view(1, 1, size, size)

    @classmethod
    def gabor_kernel(cls, kernel_size, frequency=0, theta=0, phase=0, alpha=0.05, **kwargs):
        """
        Create a Gabor filter kernel in PyTorch.

        Args:
            kernel_size (int): kernel size.
            frequency (float): Spatial frequency of the Gabor filter.
            theta (float): Orientation of the Gabor filter in radians.
            phase (float): Phase offset.
            alpha (float): Significance level for determining the z-score.
            sigma (float): Standard deviation of the Gaussian envelope = (ks-1)/2 * invcdf(1-alpha/2)
            device (str or torch.device): Device to create the tensor on.

        Returns:
            torch.Tensor: Gabor filter kernel.
        """
        # Convert alpha to a z-score using the inverse Gaussian CDF
        from scipy.stats import norm
        z_score = norm.ppf(1 - alpha / 2)
        
        # Calculate kernel size based on sigma and z_score. Ensure ks is odd.
        sigma = (kernel_size-1) / 2 * z_score 
        
        xv, yv = torch.meshgrid(
            [torch.linspace(-(kernel_size//2), kernel_size//2, steps=kernel_size)]*2, 
            indexing='ij'  # Specify indexing='ij' for Cartesian indexing
        )
        xv = xv.float()
        yv = yv.float()

        # Rotation for Gabor-specific operations
        theta_tensor = torch.tensor(theta)
        x_theta = xv * torch.cos(theta_tensor) + yv * torch.sin(theta_tensor)
        y_theta = -xv * torch.sin(theta_tensor) + yv * torch.cos(theta_tensor)

        # Gaussian envelope
        kernel = torch.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2) * torch.cos(2 * np.pi * frequency * x_theta + phase)

        # Normalize the kernel (especially important for Gaussian to ensure it sums to 1)
        kernel /= kernel.sum()

        return kernel.view(1, 1, kernel_size, kernel_size)

    @classmethod
    def get_tensor_conv2d_moments(cls, tensor, kernel_size, stride, dilation, eps=1e-8, **kwargs):
        global timesteps
        
        #print("conv", kwargs)
        B, channels, height, width = tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)
        if kernel_size is None:
            kernel_size1, kernel_size2 = height, width
            padding = 0
        else:
            kernel_size1, kernel_size2 = kernel_size, kernel_size
            padding = kernel_size // 2

        # Assuming timesteps is defined and has shape [B]
        def ts_scale(min_val, max_val):
            return 1
            # if min_val < max_val:
            #     return min_val + (max_val - min_val) * timesteps / 1000
            # else:
            #     return max_val + (max_val - min_val) * timesteps / 1000

        channel_weights = torch.tensor(kwargs.get("cw", [1,1,1,1]), device=tensor.device, dtype=torch.float32).repeat(B, 1)
        adjustment_factors = [
            torch.ones_like(timesteps) * (ts_scale(0.9, 1.1) if kernel_size and kernel_size <= 3 else 1),
            torch.ones_like(timesteps) * (ts_scale(0.9, 1.1) if kernel_size and kernel_size <= 3 else 1),
            torch.ones_like(timesteps) * (ts_scale(0.9, 1.1) if kernel_size and kernel_size <= 3 else 1),
            torch.ones_like(timesteps) * (ts_scale(1.1, 0.9) if kernel_size and kernel_size <= 3 else 1),
        ]
        # Correctly apply the adjustment factor to the last channel
        # No need to reshape adjustment_factor to [B, -1, 1, 1] for this operation
        channel_weights *= torch.stack(adjustment_factors, dim=1).to(device=tensor.device)

        # If channel_weights needs to be in shape [B, C, 1, 1] for subsequent operations:
        channel_weights = channel_weights.view(B, -1, 1, 1) / channel_weights.mean()
        
        channel_weights = torch.clamp(channel_weights, min=eps)

        channel_weights = channel_weights.pow(0.5).detach()
        kernels = {
            "avg_ind": torch.ones((channels, 1, kernel_size1, kernel_size2), device=tensor.device) / (kernel_size1 * kernel_size2),
            #"avg_comb": torch.ones((1, channels, kernel_size1, kernel_size2), device=tensor.device) / (channels * kernel_size1 * kernel_size2),
        }
        if kernel_size1 % 2 == 1 and kernel_size1 > 1 and kernel_size1 < min(tensor.shape[2],tensor.shape[3])/10:
            kernels["gaussian"] = cls.gabor_kernel(kernel_size1).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            kernels["gabor"] = cls.gabor_kernel(kernel_size1, **kwargs).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f7"] = cls.gabor_kernel(kernel_size1, 2**-7, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f6"] = cls.gabor_kernel(kernel_size1, 2**-6, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f5"] = cls.gabor_kernel(kernel_size1, 2**-5, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f4"] = cls.gabor_kernel(kernel_size1, 2**-4, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f3"] = cls.gabor_kernel(kernel_size1, 2**-3, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f2"] = cls.gabor_kernel(kernel_size1, 2**-2, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_f1"] = cls.gabor_kernel(kernel_size1, 2**-1, kwargs["theta"], kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a0"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 0*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a1"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 1*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a2"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 2*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a3"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 3*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a4"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 4*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a5"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 5*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a6"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 6*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_a7"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], 7*np.pi/8, kwargs["phase"]).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_p0"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 0*np.pi/8).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # #kernels["gabor_p1"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 1*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_p2"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 2*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # #kernels["gabor_p3"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 3*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_p4"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 4*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # #kernels["gabor_p5"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 5*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # kernels["gabor_p6"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 6*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
            # #kernels["gabor_p7"] = cls.gabor_kernel(kernel_size1, kwargs["frequency"], kwargs["theta"], 7*np.pi/4).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)
        # if kernel_size1 == 3:
        #     kernels["laplacian"] = torch.tensor([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1).to(tensor.device, tensor.dtype)

        # [0.01, 0.05, 0.1, 0.2], 
        # [1, 5], 
        # [0, (1/4)*np.pi, (1/2)*np.pi, (3/4)*np.pi],
        # [0, (1/2)*np.pi]

        # def compute_focal_weight(x, mean=0, var=1, alpha, gamma)
        #     def compute_pdf(x, mean=0, var=1):
        #         """Compute the Gaussian PDF and raise it to the power of gamma."""
        #         eps = 1e-8  # A small epsilon to prevent division by zero
        #         var = torch.clamp(var, min=eps)
        #         pdf = torch.exp(-0.5 * ((x - mean) ** 2) / var) / torch.sqrt(2 * torch.pi * var)
        #         pdf = torch.pow(pdf, gamma)  # Raise the PDF to the power of gamma
        #         return pdf
        #     pdf = F.conv2d(compute_pdf(tensor), kernel, padding=padding, stride=stride, dilation=dilation, groups=kernel.shape[0])

        if kwargs.get("kernel_type"):
            kernels = {k:v for k,v in kernels.items() if k == kwargs.get("kernel_type")}
        #print(kwargs.get("kernel_type"), kernels)

        mean_list, var_list, skew_list, kurt_list, n_list = [], [], [], [], []

        for mode, kernel in kernels.items():

            mean = F.conv2d(tensor, kernel, padding=padding, stride=stride, dilation=dilation, groups=kernel.shape[0])
            n = kernel.shape[1] * kernel.shape[2] * kernel.shape[3] * torch.ones_like(mean)
            #print(mean)
            if ("gaussian" in mode or "gabor" in mode or "avg" in mode) and kernel.shape[1]*kernel.shape[2]*kernel.shape[3] > 1:
                m2 = F.conv2d(tensor.pow(2), kernel, padding=padding, stride=stride, dilation=dilation, groups=kernel.shape[0])
                var = m2 - mean.pow(2) + eps
                m3 = F.conv2d(tensor.pow(3), kernel, padding=padding, stride=stride, dilation=dilation, groups=kernel.shape[0])
                skew = (m3 - 3 * mean * var - mean.pow(3)) / (var.pow(1.5) + eps)
                m4 = F.conv2d(tensor.pow(4), kernel, padding=padding, stride=stride, dilation=dilation, groups=kernel.shape[0])
                kurt = (m4 - 4 * mean * m3 + 6 * mean.pow(2) * m2 - 3 * mean.pow(4)) / (var.pow(2) + eps) -1 #- 3
                kurt = torch.clamp(kurt, 0.1)
            elif "laplacian" in mode:
                laplacian = mean
                mean = laplacian.mean(dim=[2, 3], keepdim=True)
                var = laplacian.var(dim=[2, 3], keepdim=True)
                #skew = torch.ones_like(mean)
                #kurt = torch.ones_like(mean)
                standardized_laplacian = (laplacian - mean) / (var.sqrt() + eps)
                third_power = torch.pow(standardized_laplacian, 3)
                fourth_power = torch.pow(standardized_laplacian, 4)
                skew = third_power.mean(dim=[2, 3], keepdim=True)
                kurt = fourth_power.mean(dim=[2, 3], keepdim=True) - 1
                mean = mean.repeat(1,1,tensor.shape[2],tensor.shape[3])/(tensor.shape[2]*tensor.shape[3]) + eps
                var = var.repeat(1,1,tensor.shape[2],tensor.shape[3])/(tensor.shape[2]*tensor.shape[3]) + eps
                skew = skew.repeat(1,1,tensor.shape[2],tensor.shape[3])/(tensor.shape[2]*tensor.shape[3]) + eps
                kurt = kurt.repeat(1,1,tensor.shape[2],tensor.shape[3])/(tensor.shape[2]*tensor.shape[3]) + eps
            else:
                var = torch.ones_like(mean)  # Variance set to eps for numerical stability
                skew = torch.ones_like(mean)
                kurt = torch.ones_like(mean)

            if "gabor" in mode:
                mean = torch.where(mean > 0, torch.log(mean), -torch.log(torch.abs(mean)))
                var = torch.ones_like(mean)  # Variance set to eps for numerical stability
                skew = torch.ones_like(mean)
                kurt = torch.ones_like(mean)
                #var = torch.log(var)
                #skew = torch.where(skew > 0, torch.log(skew), -torch.log(torch.abs(skew)))
                #kurt = torch.log(kurt)
            #print(mean.shape, var.shape, skew.shape, kurt.shape) 
            if mean.shape[1] == 40000:
                mean_list.append(torch.clamp(mean*channel_weights, min=eps))
                var_list.append(torch.clamp(var*channel_weights, min=eps))
                skew_list.append(torch.clamp(skew*channel_weights, min=eps))
                kurt_list.append(torch.clamp(kurt*channel_weights, min=eps))
            else:
                mean_list.append(mean)
                var_list.append(var)
                skew_list.append(skew)
                kurt_list.append(kurt)
            n_list.append(n)

        return torch.cat(mean_list, dim=1), torch.cat(var_list, dim=1), torch.cat(skew_list, dim=1), torch.cat(kurt_list, dim=1), torch.cat(n_list, dim=1)

    @classmethod
    def focal_weight(cls, true_vals, true_var, pred_vals, pred_var, sample_num, dist_mean=0, dist_var=1, focal_gamma=0.0005):
        """
        Upweight hard to predict values that we are not predicting well
        """
        dist_mean, dist_var = dist_mean * torch.ones_like(true_vals), dist_var * torch.ones_like(true_var)

        def joint_p_val(pv1, pv2):
            # Use Fisher's method to compute joint p-value from two independent p-values
            combined_statistic = -2 * (pv1.log() + pv2.log())
            p_value = 1 - torch.distributions.Chi2(df=torch.tensor(4, device=combined_statistic.device)).cdf(combined_statistic)
            return p_value

        def p_value(sample_mean, sample_var, sample_num, pop_mean, pop_var):
            # Compute mean pvalue: prob we observe mean1 | N(mean2, var2)
            z_score = -(sample_mean - pop_mean).abs() / (pop_var)**0.5
            p_value = (1 + torch.erf(z_score * 2**-0.5 )) # 2-tailed test p-value for target for theoretical N(0, 1)
            sample_num = torch.tensor(sample_num, device=sample_mean.device) if not torch.is_tensor(sample_num) else sample_num
            # if not (sample_num == 1).any():
            #     # Compute var pvalue: prob we observe var1 | N(mean2, var2)
            #     p_value_sample_var = 1 - torch.distributions.Chi2(df=n-1).cdf((n-1) * (sample_var + 1e-4) / (pop_var + 1e-4))
            #     # Use Fisher's Method to compute joint p-value, these are independent per CLT
            #     p_value = joint_p_val(p_value, p_value_sample_var)
            return p_value

        """
        Compute P(noise | N(0, 1))
        How unlikely was this noise, these will be harder for model to reproduce
        """
        prob_true_given_dist = p_value(true_vals, true_var, sample_num, dist_mean, dist_var)

        """
        Compute P(noise_pred | N(noise_mean, noise_var))
        How unlikely was this predicted - I think KL-Divergence might already take this into account (u1-u2)^2 / var2 = z_score^2
        """
        #prob_pred_given_true = p_value(pred_vals, pred_var, sample_num, true_vals.mean(dim=-1), true_var.var(dim=-1))

        # Use Fisher's method to compute joint p-value 
        #p_value = joint_p_val(prob_pred_given_true, prob_true_given_dist)
        #p_value = (prob_pred_given_true + 7*prob_true_given_dist) / 8
        #p_value = (prob_true_given_dist * prob_pred_given_true) ** 0.5
        p_value = prob_true_given_dist
        #focal_gamma = 0.0005
        #gamma = 0.005
        if torch.is_tensor(focal_gamma):
            for _ in range(p_value.dim() - focal_gamma.dim()):
                focal_gamma = focal_gamma.unsqueeze(-1)

        alpha = 0.5**focal_gamma # stabilize weight to be 1 for in the middle, p-val will be uniform over null hypothesis
        # **0.5
        #focal_weight = 1 + alpha - p_value.pow(gamma)
        focal_weight = 1 + focal_gamma * ( 1 / (p_value + 1e-3) - 1)
        #focal_weight = torch.ones_like(focal_weight)
        """
        gamma = 2       gamma = 0.1
        alpha = 0.25    alpha = 0.93
        p_value | w | g=2   | w | g=0   | w | g=0.1
        --------|-----------|-----------|--------
        1       | 0.25      | 1         |   0.93
        0.5     | 1         | 1         |   1
        0.25    | 1.19      | 1         |   1.06
        0.1     | 1.24      | 1         |   1.14
        0       | 1.25      | 1         |   1.93
        so, unlikely cases will have their loss upweighted 5x relative to easy cases

        """
        #return torch.ones_like(focal_weight)
        return focal_weight.detach()


    @classmethod
    def kl_divergence(cls, tensor1, tensor2, kernel_size=None, stride=1, dilation=1, eps=1e-4, mode='independent', **kwargs):
        # Compute means and variances using convolutions
        img_dims = min(tensor1.shape[2], tensor1.shape[3])
        kwargs["frequency"] = np.random.choice(np.geomspace(start=0.01, stop=0.5, num=100))
        kwargs["theta"] = np.random.uniform(0, np.pi)
        kwargs["phase"] = np.random.uniform(0, 2 * np.pi)

        mean1, var1, skew1, kurt1, n1 = cls.get_tensor_conv2d_moments(tensor1, kernel_size, stride, dilation, eps, **kwargs)
        mean2, var2, skew2, kurt2, n2 = cls.get_tensor_conv2d_moments(tensor2, kernel_size, stride, dilation, eps, **kwargs)

        def kl_diverge(mean1, var1, mean2, var2, n=1, focal_loss=False):
            focal_weight = cls.focal_weight(mean1, var1, mean2, var2, n) if focal_loss else 1
            kl_div = (0.5 * ((var2 / var1).log() + (var1 + (mean1 - mean2).pow(2)) / var2 - 1))
            return focal_weight * kl_div

        def js_diverge(mean1, var1, mean2, var2):
            M_mean, M_var = 0.5 * (mean1 + mean2), 0.5 * (var1 + var2)
            kl_pm = kl_diverge(mean1, var1, M_mean, M_var)
            kl_qm = kl_diverge(mean2, var2, M_mean, M_var)            
            return 0.5 * (kl_pm + kl_qm)
        
        # Compute KL Divergence
        kl_div = kl_diverge(mean1, var1, mean2, var2, n1, focal_loss=True)
        #kl_div += kl_diverge(skew1, kurt1**0.5, skew2, kurt2**0.5)

        # Compute JS Divergence
        # kl_div += js_diverge(mean1, var1, mean2, var2)
        # kl_div += js_diverge(skew1, kurt1**0.5, skew2, kurt2**0.5)
        
        # if inf_mask.any():
        #     print("inf values detected in KL divergence calculation.")
            
        #     # Print matching elements from kurt1 and kurt2
        #     print("kurt1 elements leading to inf:", kurt1[inf_mask])
        #     print("kurt2 elements leading to inf:", kurt2[inf_mask])
        #     print("log elements leading to inf:", torch.log(kurt2 / kurt1)[inf_mask])
        #     print("ratio elements leading to inf:", (kurt1/kurt2)[inf_mask])
    
        kl_div = kl_div.mean(dim=[1, 2, 3])
        #print(kl_div.mean())

        # normalize to MSE scale
        return 2*kl_div 

    @classmethod
    def pyramid_loss(cls, tensor1, tensor2, func, kernel_range=[1, 200], steps=20, factor=None, **kwargs):
        min_kernel_size, max_kernel_size = kernel_range
        image_size = min(tensor1.size(2), tensor1.size(3))  # Assuming BCHW format
        #print("pl", kwargs)
        def generate_image_pyramid(tensor):
            pyramid = [tensor]
            # for level in range(1, math.floor(math.log(image_size/max_kernel_size, 2)) ):            
            #     # Use the computed scale directly with interpolate
            #     scaled_tensor = F.interpolate(tensor, scale_factor=2 ** -level, mode='bilinear', align_corners=False)
            #     pyramid.append(scaled_tensor)
            return pyramid

        # Calculate scale_steps or scale_factor if one is given
        if steps == -1:
            scales = [{"scale": scale, "count": 1} for scale in list(range(min_kernel_size, max_kernel_size + 1))] # + [image_size]]
        else:
            if factor is None:
                scale_factor = np.power(image_size, 1/steps)
            else:
                steps = np.log(image_size) / np.log(factor)
                scale_factor = factor

            # Generate scales
            #rounded_scales = np.round(np.geomspace(1, image_size, num=steps)).astype(int)
            rounded_scales = list(map(lambda x: round(x) if x <= 6 else (math.floor(x) if math.floor(x) % 2 == 1 else math.ceil(x)), list(np.geomspace(1, image_size, num=steps))))

            # Count occurrences of each rounded scale
            from collections import Counter
            counts = Counter(rounded_scales)

            scales = [{"scale": scale, "count": count} for scale, count in counts.items()]
        steps = len(scales)

        for rank, scale in enumerate(scales):
            scale["rank"] = rank
            if scale["scale"] == image_size:
                scale["kernel_size"] = None
                scale["dilation"] = 1
            elif scale["scale"] <= max_kernel_size:
                scale["kernel_size"] = int(np.round(scale["scale"])) if scale["scale"] != image_size else None
                scale["dilation"] = 1
            else:
                # For scales larger than max_kernel_size, adjust dilation instead
                scale["kernel_size"] = max_kernel_size
                scale["dilation"] = int(np.ceil(scale["scale"] / scale["kernel_size"]))
            
            scale["stride"] = max(1, int(scale["kernel_size"]*scale["dilation"] / 4)) if scale["kernel_size"] else 1 # Example stride calculation
            scale["weight"] = scale["count"] * (1 - scale["rank"]/(steps*4)) * (max(1 + scale["rank"], steps - scale["rank"]) /(steps/2) - (1 - 1/steps))
            scale["weight"] = 1

            # Example weight calculation; should be refined based on specific needs

        # Generate image pyramids
        pyramid1 = generate_image_pyramid(tensor1)
        pyramid2 = generate_image_pyramid(tensor2)

        losses = []
        weights = []
        for level in range(0, len(pyramid1)):
            level_weight = 1 / (level+1)
            for scale in scales:
                if scale["scale"] > min(pyramid1[level].size(2), pyramid1[level].size(3)):
                    continue
                kwargs.update(scale)
                weight = scale["weight"] * level_weight
                if scale["kernel_size"] in [1, 2, None]:
                    weight *= 2
                loss = weight * func(pyramid1[level], pyramid2[level], **kwargs) 
                scale["loss"] = loss
                losses.append(loss)
                weights.append(weight)

            # for scale in scales:
            #     #scale["loss"]
            #     scale["weight"] /= sum(weights)
            #     print(scale)

        return sum(losses) / sum(weights)

    @staticmethod
    def clipped_latents(tensor, boundary=1 / 0.13025 / 2):
        # Assuming the tensor shape is [batch, width, height, channel]
        # If it's [batch, channel, width, height], no change is needed in the code

        # Step 1: Identify out-of-range elements
        out_of_range = (tensor < -boundary) | (tensor > boundary)

        # Step 2: Count out-of-range elements for each item in the batch
        out_of_range_count = torch.sum(out_of_range.view(tensor.shape[0], -1), dim=1)

        # Step 3: Total number of elements in each item of the batch
        total_elements = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]

        # Step 4: Compute percentage
        percentage = (out_of_range_count.float() / total_elements) 

        return 100*percentage


    @classmethod
    def sharpness_tensor(cls, images_tensor, mode='average'):
        if images_tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional [batch_size, channels, height, width]")

        num_channels = images_tensor.shape[1]

        # Define a Laplacian kernel
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]], dtype=images_tensor.dtype, device=images_tensor.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

        if mode == 'separate':
            sharpness_values_per_channel = []

            for i in range(num_channels):
                # Extract the single channel
                single_channel = images_tensor[:, i:i+1, :, :]

                # Apply the Laplacian filter
                laplacian = F.conv2d(single_channel, laplacian_kernel, padding=1)

                # Compute the standard deviation for each image in the channel
                channel_sharpness = laplacian.view(laplacian.shape[0], -1).std(dim=1)
                sharpness_values_per_channel.append(channel_sharpness)

            # Combine sharpness values from all channels
            combined_sharpness = torch.stack(sharpness_values_per_channel, dim=1).mean(dim=1)

        elif mode == 'average':
            # Convert to grayscale by averaging the channels
            grayscale = images_tensor.mean(dim=1, keepdim=True)

            # Apply the Laplacian filter
            laplacian = F.conv2d(grayscale, laplacian_kernel, padding=1)

            # Compute the standard deviation for each image in grayscale
            combined_sharpness = laplacian.view(laplacian.shape[0], -1).std(dim=1)

        else:
            raise ValueError("Invalid mode. Choose 'separate' or 'average'.")

        return combined_sharpness



def call_unet_text(network, args, accelerator, unet, noisy_latents, timesteps, text_modifier, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder):
    
    # Function to pad tensor
    def pad_tensor(tensor, pad_value, target_length=77):
        padding_needed = target_length - tensor.size(-1)
        padded_tensor =  F.pad(tensor, (0, padding_needed), value=pad_value)
        return padded_tensor

    # Backup the input_id tokens to pass in batch
    input_ids = batch["input_ids"]

    # Add text conditioning to current caption
    if text_modifier != "":
        text_modifier += ", "
    captions = [text_modifier + f for f in batch["captions"]]

    # Process based on the number of tokenizers
    if len(tokenizers) == 1:
        # Tokenize the captions with a single tokenizer
        batch["input_ids"] = tokenizers[0](captions, padding=True, truncation=True, return_tensors="pt").input_ids.to(accelerator.device).unsqueeze(1)
        # Pad to target length if needed
        batch["input_ids"] = pad_tensor(batch["input_ids"], tokenizers[0].pad_token_id, tokenizers[0].model_max_length)
    else:
        # Tokenize the captions with both tokenizers and pad accordingly
        batch["input_ids"] = tokenizers[0](captions, padding=True, truncation=True, return_tensors="pt").input_ids.to(accelerator.device).unsqueeze(1)
        batch["input_ids2"] = tokenizers[1](captions, padding=True, truncation=True, return_tensors="pt").input_ids.to(accelerator.device).unsqueeze(1)
        batch["input_ids"] = pad_tensor(batch["input_ids"], tokenizers[0].pad_token_id, tokenizers[0].model_max_length)
        batch["input_ids2"] = pad_tensor(batch["input_ids2"], 0, tokenizers[1].model_max_length)

        # Backup the second input_ids to restore later
        input_ids2 = batch["input_ids2"]

    # Generate hidden encoding layers
    with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
        text_encoder_conds = network.get_text_cond(
            args, accelerator, batch, tokenizers, text_encoders, weight_dtype
        )

    # Call the UNet with the text encoder conditioning
    noise_pred = network.call_unet(
        args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype
    )

    # Restore batch input_ids
    batch["input_ids"] = input_ids
    if len(tokenizers) > 1:
        batch["input_ids2"] = input_ids2  # Restore only if it exists
    noise_pred.requires_grad = False
    
    return noise_pred

def weighted_mean_noise(network, args, accelerator, unet, noisy_latents, currenttimesteps, text_modifiers, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder):
    global timesteps
    timesteps = currenttimesteps
    
    if len(text_modifiers) > 1 and text_guidance_scale > 0:
        text_modifiers = {k:v for k,v in [random.choice(list(text_modifiers.items()))]}

    with torch.no_grad():
        # Calculate the total weight
        weight_tot = sum(weight for _, (weight, _, _) in text_modifiers.items())

        # Initialize tensor for the weighted sum
        weighted_mean = 0

        # Loop through each entry in text_positives
        for _, (weight, text_modifier, valid_keyword_list) in text_modifiers.items():
            result = call_unet_text(
                network, args, accelerator, unet, noisy_latents, timesteps, text_modifier, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder
            )
            weighted_mean += (weight / weight_tot) * result
        return weighted_mean

def gabor_filter_stacked(tensor, frequencies, sigmas, thetas, phases, alpha=0.05, device='cpu'):
    """
    Apply Gabor filters with multiple frequencies, sigmas, angles, and phases to the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W].
        frequencies (list of float): List of spatial frequencies for the Gabor filters.
        sigmas (list of float): List of standard deviations for the Gaussian envelope.
        thetas (list of float): List of orientations of the Gabor filters in radians.
        phases (list of float): List of phase offsets.
        alpha (float): Significance level for determining the z-score.
        device (str or torch.device): Device to create the tensor on.

    Returns:
        torch.Tensor: The result of applying the stacked Gabor filter to the tensor, with expanded channels.
    """
    def create_gabor_kernel(frequency, sigma, theta, phase, ks, device='cpu'):
        xv, yv = torch.meshgrid([torch.linspace(-ks//2, ks//2, ks)], indexing='ij')
        xv, yv = torch.stack(np.meshgrid(torch.linspace(-ks//2, ks//2, ks), torch.linspace(-ks//2, ks//2, ks)), 0)
        xv, yv = xv.to(device), yv.to(device)
        theta_tensor = torch.tensor(theta, device=device)
        x_theta = xv * torch.cos(theta_tensor) + yv * torch.sin(theta_tensor)
        y_theta = -xv * torch.sin(theta_tensor) + yv * torch.cos(theta_tensor)

        gb = torch.exp(-0.5 * (x_theta ** 2 + y_theta ** 2) / sigma ** 2) * torch.cos(2 * np.pi * frequency * x_theta + phase)
        return gb.unsqueeze(0)

    # Determine the kernel size using the largest sigma
    from scipy.stats import norm
    largest_sigma = max(sigmas)
    z_score = 1.96  # Approximation for alpha=0.05
    ks = int(2 * math.ceil(z_score * largest_sigma) + 1)

    kernels = []
    for frequency in frequencies:
        for sigma in sigmas:
            for theta in thetas:
                for phase in phases:
                    kernel = create_gabor_kernel(frequency, sigma, theta, phase, ks, device)
                    kernels.append(kernel)

    # Stack all kernels and expand dimensions for convolution
    stacked_kernels = torch.cat(kernels, dim=0).unsqueeze(1)  # [N*Kernels, 1, H, W]
    
    # Reshape input tensor to apply each kernel to each channel
    B, C, H, W = tensor.shape
    input_reshaped = tensor.view(B * C, 1, H, W)  # Treat each channel as separate batch
    
    # Apply convolution
    output = F.conv2d(input_reshaped, stacked_kernels, padding=ks//2)
    
    # Reshape output back to original batch format with expanded channels
    output_reshaped = output.view(B, -1, H, W)  # [B, C*N*Kernels, H, W]
    
    return output_reshaped

def tensor_func_stack(tensor, func, kwargs_list):
    """
    Apply multiple Gabor filters to the input tensor, each with different parameters,
    and stack the results along the channel dimension.
    
    Args:
    - tensor (torch.Tensor): Input tensor of shape [B, C, H, W].
    - kwargs_list (list of dicts): List of keyword arguments for each Gabor filter call.
    
    Returns:
    - torch.Tensor: The stacked result of applying each Gabor filter to the tensor.
    """
    filtered_tensors = []
    for kwargs in kwargs_list:
        filtered = func(tensor, **kwargs)
        filtered_tensors.append(filtered)
    
    # Stack along the channel dimension (dim=1)
    return torch.cat(filtered_tensors, dim=1)

def gabor_filter(tensor, **kwargs):
    """
    Apply a Gabor filter to the input tensor.
    tensor: Input tensor of shape [B, C, H, W].
    Other parameters are passed to the gabor_kernel function.
    """
    kernel_size=2
    kernel = NoiseAnalysis.gabor_kernel(kernel_size, **kwargs, device=tensor.device)
    # kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    print(kernel.shape)  # Debug: Ensure shape is [1, 1, kernel_size, kernel_size]

    kernel = kernel.repeat(tensor.size(1), 1, 1, 1)  # Repeat kernel for each input channel
    
    # Ensure padding maintains output size
    padding = kernel.shape[-1] // 2
    kernel = kernel.to(device=tensor.device)
    
    filtered = F.conv2d(tensor, kernel, padding=padding, groups=tensor.size(1))
    return filtered

def transpose_stack(tensor):
    """
    Stack a BCHW tensor with its transpose along the channel dimension.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W].
    
    Returns:
        torch.Tensor: Tensor stacked with its transpose along the channel dimension.
    """
    # Transpose H and W dimensions: [B, C, H, W] -> [B, C, W, H]
    transposed_tensor = tensor.permute(0, 1, 3, 2)

    # Determine max dimensions for padding
    max_h = max(tensor.shape[2], transposed_tensor.shape[2])
    max_w = max(tensor.shape[3], transposed_tensor.shape[3])

    # Pad the tensors to have the same dimensions
    padded_tensor = torch.nn.functional.pad(tensor, pad=(0, max_w - tensor.shape[3], 0, max_h - tensor.shape[2]), mode='constant', value=0)
    padded_transposed_tensor = torch.nn.functional.pad(transposed_tensor, pad=(0, max_w - transposed_tensor.shape[3], 0, max_h - transposed_tensor.shape[2]), mode='constant', value=0)

    # Stack along the channel dimension
    stacked_tensor = torch.cat([padded_tensor, padded_transposed_tensor], dim=1)
    
    return stacked_tensor

def fft_stack(tensor):
    tensor = tensor[:, :3, :, :]  # Shape: [B, 3, H, W]
    fft_tensor = torch.fft.rfft2(tensor, dim=(2, 3))
    fft_tensor_shifted = torch.fft.fftshift(fft_tensor, dim=(2, 3))

    # Stack real, imaginary, amplitude, and phase components along the channel dimension
    fft_components_stacked = torch.cat([
        #fft_tensor_shifted.real, 
        #fft_tensor_shifted.imag, 
        torch.abs(fft_tensor_shifted), 
        torch.angle(fft_tensor_shifted),
    ], dim=1)
    
    return fft_components_stacked

def multivariate_kl_diverg5ence_grid(tensor1_unchunked, tensor2_unchunked, num_chunks_sqrt=None, stride_scale=0.8, transform_type=None):
    global timesteps
    B, C, H, W = tensor1_unchunked.shape

    # Perturb input arguments so training gets more varied slices/foldings for KL-divergence
    # stride_scale=0.8
    # stride_scale = min(1, max(0, stride_scale + torch.rand(1).item() * 0.2)) # +/- 0.1
    num_chunks_sqrt = int(round(((H + W)/2)**0.5))
    num_chunks_sqrt += torch.randint(-2, 2+1, (1,)).item() # + 0-2

    def chunk_and_stack(tensor, N=3, stride_scale=0.8):
        """
        Chunk and stack tensor into grids of dim [B, C, Unfold_Group_H, Unfold_Group_W, chunk_size, chunk_size]

        matrix operations scale quadratically so we break tensor into NxN chunks and stack along channel dim
        N = 3 means split into roughly 3x3 chunks
        stride_scale = 0.8 means the stride is 80% of kernel, so we have some overlap to avoid discontinuities
        """
        B, C, H, W = tensor.shape
        chunk_size = min(H, W) // N
        stride = int(chunk_size * stride_scale)

        # Use unfold to extract patches
        patches = F.unfold(tensor, kernel_size=(chunk_size, chunk_size), stride=(stride, stride))
        # Calculate number of patches (unfold groups) along H and W
        Unfold_Group_H = (H - chunk_size) // stride + 1
        Unfold_Group_W = (W - chunk_size) // stride + 1

        # Reshape to [B, C, chunk_size*chunk_size, Unfold_Group_H*Unfold_Group_W]
        patches = patches.view(B, C, chunk_size, chunk_size, Unfold_Group_H, Unfold_Group_W)

        # Permute to get [B, C, Unfold_Group_H, Unfold_Group_W, chunk_size, chunk_size]
        patches = patches.permute(0, 1, 4, 5, 2, 3)

        return patches

    def reshape_for_within_groups(tensor):
        B, C, Unfold_Group_H, Unfold_Group_W, H_chunk, W_chunk = tensor.shape
        # Flatten Unfold_Group_H and Unfold_Group_W into the channel dimension
        return tensor.reshape(B, C * Unfold_Group_H * Unfold_Group_W, H_chunk, W_chunk)

    def reshape_for_between_groups(tensor, func=torch.mean):
        # Compute mean across the chunk dimensions (H_chunk, W_chunk) for each group
        return func(tensor, dim=[-2, -1])

    def compute_spatial_distances(H, W):
        # Create a grid of coordinates (x, y) for each pixel
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        # Flatten the coordinates to vectors
        y_coords_flat = y_coords.flatten()
        x_coords_flat = x_coords.flatten()
        
        # Compute distances between all pairs of points
        distances = torch.sqrt((y_coords_flat[:, None] - y_coords_flat[None, :]) ** 2 +
                                (x_coords_flat[:, None] - x_coords_flat[None, :]) ** 2)
        return distances

    def get_mean_and_covar(tensor, stabilize_factor):
        B, C, H, W = tensor.shape
        k = H*W

        elems = tensor.view(B, C, -1, 1)
        mean = elems.mean(dim=2, keepdim=True)
        centered_elems = (elems - mean)
        
        # For variance use a 95% UB, for covariance use a 95% LB
        from scipy.stats import chi2
        var = (centered_elems**2).sum(dim=2, keepdim=True) / chi2.ppf(0.05/2, k-1)
        cov_matrices = torch.matmul(centered_elems, centered_elems.transpose(-2, -1)) / (k-1) 

        # Stabilize the diagonals
        I = torch.eye(k).to(tensor.device) 
        cov_matrices += I * stabilize_factor

        # shrinkage_alpha = 0.99
        # cov_matrices = (1-shrinkage_alpha) * cov_matrices + shrinkage_alpha * I * var / k

        # Use spatial correlation to downscale off diagonal elements of the covariance matrix by their distance
        # banding_min, banding_scale = 0.5, 0.5
        # distances = compute_spatial_distances(H, W).to(tensor.device)
        # #banding_mask = (distances <= banding_threshold).float()
        # banding_mask =  banding_min + (1-banding_min) / (1+distances) ** banding_scale
        # cov_matrices *= banding_mask

        return elems, mean, cov_matrices,  var 

    def kl_divergence(purpose, tensor1, tensor2, agg_obs=1):
        B, C, H, W = tensor1.shape

        stabilize_factor = tuning_factors[purpose]["stabilize_factor"]
        mse_weight = tuning_factors[purpose]["mse_weight"]
        trace_weight = tuning_factors[purpose]["trace_weight"]
        logdet_weight = tuning_factors[purpose]["logdet_weight"]
        covar_tune = tuning_factors[purpose]["covar_tune"]
        
        total_weight = tuning_factors[purpose]["total_weight"](agg_obs)
        dist_mean, dist_var = tuning_factors[purpose]["dist"](agg_obs)

        u1, mean1, cov1, var1 = get_mean_and_covar(tensor1, stabilize_factor)
        u2, mean2, cov2, var2 = get_mean_and_covar(tensor2, stabilize_factor)            

        focal_gamma = tuning_factors[purpose]["focal_gamma"] 
        # if focal_gamma > 0:
        #     focal_gamma *= (var1/var2)

        k_term = torch.tensor(H*W, device=tensor1.device)

        weight = NoiseAnalysis.focal_weight(u1, var1, u2, var2, k_term, dist_mean=dist_mean, dist_var=dist_var, focal_gamma=focal_gamma).view(B, C, H*W, 1)
        # if purpose == "FFT":
        #     weight = torch.ones_like(weight)

        #var_weight = NoiseAnalysis.focal_weight(u1, var1, u2, var2, k_term, dist_var=1/agg_obs).view(B, C, H*W, 1)

        cov2_inv = cov2.cholesky_inverse(upper=False)

        #mse_term = total_weight * stabilize_factor * torch.matmul(torch.matmul((u2-u1).transpose(-1,-2),cov2_inv), (u2-u1)).view(B, C) / k_term 

        # MSE: SUM((pred-label)^2) / var(pred)
        mse_term = mse_weight * total_weight * stabilize_factor * torch.matmul(torch.matmul((u2-u1).transpose(-1,-2),cov2_inv), weight * (u2-u1)).view(B, C) / k_term 
        #trace_term = total_weight * (stabilize_factor * torch.einsum('bcij,bcjk->bc', cov2_inv, cov1) / k_term - 1)  

        # Trace: var(label) / var(pred) - 1
        trace_term = trace_weight * total_weight * stabilize_factor * ((weight.squeeze() * torch.einsum('bcij,bcjk->bci', cov2_inv, cov1)).sum(dim=-1) / k_term - 1) # weight trace diagonals before summing them

        # LogDet: Cov(pred) / Cov(label)
        logdet_term = logdet_weight * total_weight * stabilize_factor * (cov2.logdet() - covar_tune*cov1.logdet()) / k_term  # multiplying eps * cov2.logdet() increased saturation?
        
        # Compute cross covariance given we have extra knowledge that u1 and u2 would ideally be perfectly correlated, not just from same distribution

        # ideally this is as high as possible?
        #print("corr.shape", (torch.matmul(u2-mean2, (u1-mean1).transpose(-1,-2)) / (var1*var2)*0.5 / k_term ).shape)
        # correlation_matrix = total_weight * (torch.matmul((u1-mean1).transpose(-1,-2), u2-mean2) / (var1*var2)*0.5 / k_term ) #.sum(dim=[-1, -2])
        
        # # Diagonals are good
        # diagonals = (torch.diagonal(correlation_matrix, dim1=-2, dim2=-1)**2 ).mean(dim=-1)

        # # off diagonals should match
        # difference_matrix = correlation_matrix - correlation_matrix.transpose(-2, -1)
        # # Create a mask to exclude diagonal elements (set them to 0)
        # N = correlation_matrix.size(-1)  # Assuming the last two dimensions are both N
        # mask = torch.eye(N, device=correlation_matrix.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        # difference_matrix.masked_fill_(mask, 0)

        # # Now, sum over the last two dimensions to get the sum of off-diagonal differences for each [B, C]
        # sum_off_diagonal_differences = (difference_matrix**2).mean(dim=[-2, -1])

        # print("var", var1.shape)
        # correlation = (sum_off_diagonal_differences - diagonals) / (var1 * var2).squeeze(-1).squeeze(-1)
        # print("sodd", sum_off_diagonal_differences.mean(dim=1))
        # print("diag", diagonals.mean(dim=1))

        # mse_term /= k_term*10
        # trace_term /= k_term*10 
        # logdet_term /= k_term*10 

        if torch.isnan(logdet_term).any():
            logdet_term = torch.zeros_like(logdet_term)
        # L_cov1 = torch.cholesky(cov1_reg)
        # L_cov2 = torch.cholesky(cov2_reg)
        # logdet_cov1 = 2 * torch.sum(torch.log(torch.diagonal(L_cov1, dim1=-2, dim2=-1)), dim=-1)
        # logdet_cov2 = 2 * torch.sum(torch.log(torch.diagonal(L_cov2, dim1=-2, dim2=-1)), dim=-1)


        kl_div = 0.5 * (trace_term + mse_term + logdet_term)
        
        print("{} MSE (obs)".format(purpose), ", ".join([f"{value:.5f}" for value in mse_term.mean(dim=[1]).tolist()]))
        print("{} Trace (var)".format(purpose), ", ".join([f"{value:.5f}" for value in trace_term.mean(dim=[1]).tolist()]))
        print("{} LogDet (cov)".format(purpose), ", ".join([f"{value:.5f}" for value in logdet_term.mean(dim=[1]).tolist()]))
        print("{} KL-Divergence".format(purpose), ", ".join([f"{value:.5f}" for value in kl_div.mean(dim=[1]).tolist()]))
        #print("{} Correlation".format(purpose), ", ".join([f"{value:.5f}" for value in correlation.mean(dim=[1]).tolist()]))

        return kl_div.mean(dim=1) # - correlation.mean(dim=1)

    #stabilize_factor = 0.1
    #stabilize_factor = 1

    tuning_factors = {
        "Within": {
            "stabilize_factor": 0.1,
            "mse_weight": 1,
            "trace_weight": 1,
            "logdet_weight": 1,
            "covar_tune": 1.0001,
            #"covar_tune": 1.0,
            "dist": lambda agg_obs: (0, 1/agg_obs),
            "total_weight": lambda agg_obs: 1/agg_obs,
            #"focal_gamma": 0.0,
            "focal_gamma": 0.005,
        },
        "Between Means": {
            "stabilize_factor": 0.1,
            "mse_weight": 0.1,
            "trace_weight": 0.1,
            "logdet_weight": 0.1,
            "covar_tune": 1,
            "dist": lambda agg_obs: (0, 1/agg_obs),
            "total_weight": lambda agg_obs: torch.sigmoid(0.01 * (timesteps - 500)).unsqueeze(1) / agg_obs,
            "focal_gamma": 0.0,
        },
        "Between Vars": {
            "stabilize_factor": 0.1,
            "mse_weight": 0.1,
            "trace_weight": 0.1,
            "logdet_weight": 0.1,
            "covar_tune": 1,
            "dist": lambda agg_obs: (1, 2 * 1**2 / (agg_obs-1)), # here dist_mean = sigma^2)
            "total_weight": lambda agg_obs: torch.sigmoid(0.01 * (timesteps - 500)).unsqueeze(1) / agg_obs,
            "focal_gamma": 0.0,
        }
    }



    tensor1_6d = chunk_and_stack(tensor1_unchunked, num_chunks_sqrt, stride_scale)
    tensor2_6d = chunk_and_stack(tensor2_unchunked, num_chunks_sqrt, stride_scale)
    B, C, unfold_H, unfold_W, pixel_H, pixel_W = tensor1_6d.shape
    
    #if transform_type is None:
    # KL-Divergence within each grid
    kl_div_within_chunks = kl_divergence(
        "Within",
        reshape_for_within_groups(tensor1_6d),
        reshape_for_within_groups(tensor2_6d)
    ) 

    kl_div_between_chunks_mean = 0
    kl_div_between_chunks_var = 0
    # KL-Divergence of the grid means
    kl_div_between_chunks_mean = kl_divergence(
        "Between Means",
        reshape_for_between_groups(tensor1_6d),
        reshape_for_between_groups(tensor2_6d),
        (pixel_H * pixel_W)
    )

    # KL-Divergence of the grid variances
    kl_div_between_chunks_var = kl_divergence(
        "Between Vars",
        reshape_for_between_groups(tensor1_6d, torch.var),
        reshape_for_between_groups(tensor2_6d, torch.var),
        (pixel_H * pixel_W)
    )
    kl_div = kl_div_within_chunks + kl_div_between_chunks_mean + kl_div_between_chunks_var
    # elif transform_type == "FFT":
    #     kl_div = kl_divergence(
    #         "FFT",
    #         reshape_for_between_groups(tensor1_6d),
    #         reshape_for_between_groups(tensor2_6d),
    #         (pixel_H * pixel_W)
    #     )          

    return kl_div

def pairwise_loss(tensor_pred, tensor_true=None, power=2):
    """
    Compute the pairwise MSE loss between each batch item and every other batch item in a BCHW tensor.
    
    Args:
    - tensor (torch.Tensor): Input tensor of shape (B, C, H, W)

    Returns:
    - torch.Tensor: Pairwise MSE matrix of shape (B, B), where element (i, j) 
                        represents the MSE between batch items i and j.
    """ 
    global timesteps
    B, C, H, W = tensor_pred.shape
    
    # Expand tensor to form all pairs: shape becomes (B, B, C, H, W)
    tensor_expanded_row = tensor_pred.unsqueeze(1).expand(-1, B, -1, -1, -1)
    tensor_expanded_col = (tensor_true if tensor_true is not None else tensor_pred).unsqueeze(0).expand(B, -1, -1, -1, -1)
    timestep_diffs = (timesteps.unsqueeze(1) - timesteps.unsqueeze(0)).abs()
    timestep_scale = (1 - timestep_diffs/1000).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).detach()

    # Compute squared differences: shape remains (B, B, C, H, W)
    squared_diff = ((tensor_expanded_row - tensor_expanded_col) * timestep_scale) ** power

    if power % 2 == 1:
        squared_diff = squared_diff.abs()
    
    # Compute mean over the spatial dimensions (and optionally channel dimension): resulting shape (B, B)
    mse_matrix = squared_diff.mean(dim=(3, 4))  # Include dim=2 if you want to average over channels as well
    #mse_matrix = squared_diff.mean(dim=(2, 3, 4))  # Include dim=2 if you want to average over channels as well
    #mse_matrix = torch.minimum(mse_matrix, torch.tensor(3, device=mse_matrix.device))
    #mse_matrix *= 1 if tensor_true is None else 2
    return mse_matrix

def pairwise_covariance_sum(tensor_pred, tensor_true=None, kernel_size=8):
    """
    Compute the pairwise MSE loss between each batch item and every other batch item in a BCHW tensor.
    
    Args:
    - tensor (torch.Tensor): Input tensor of shape (B, C, H, W)

    Returns:
    - torch.Tensor: Pairwise MSE matrix of shape (B, B), where element (i, j) 
                        represents the MSE between batch items i and j.
    """
    global timesteps
    B, C, H, W = tensor_pred.shape
    
    tensor_pred_reduced = F.avg_pool2d(tensor_pred, kernel_size=kernel_size)  # Example reduction

    if tensor_true is not None:
        tensor_true_reduced = F.avg_pool2d(tensor_true, kernel_size=kernel_size)
    else:
        tensor_true_reduced = tensor_pred_reduced

    # Flattening the reduced tensors to shape [B, C, N] where N = H'*W' (H' and W' are the reduced dimensions)
    N = tensor_pred_reduced.size(2) * tensor_pred_reduced.size(3)  # Calculate the product of the reduced dimensions
    tensor_pred_flat = tensor_pred_reduced.view(B, C, N)
    tensor_true_flat = tensor_true_reduced.view(B, C, N)

    # compute mean
    tensor_pred_flat_mean = tensor_pred_flat.mean(dim=2, keepdim=True)
    tensor_true_flat_mean = tensor_true_flat.mean(dim=2, keepdim=True)

    # compute centered
    tensor_pred_centered = tensor_pred_flat - tensor_pred_flat_mean
    tensor_true_centered = tensor_true_flat - tensor_true_flat_mean

    # Expand tensor to form all pairs: now aiming for shape (B, B, C, N, 1)
    # Detach one half of these?
    tensor_expanded_row = tensor_pred_centered.unsqueeze(1).unsqueeze(-1).expand(-1, B, -1, -1, -1)
    tensor_expanded_col = tensor_true_centered.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1, -1)

    cov_matrices = torch.matmul(tensor_expanded_row, tensor_expanded_col.transpose(-2, -1)) / (N-1) 

    # Create a mask to zero out diagonal elements for BxB comparisons
    mask = torch.eye(B, dtype=torch.bool, device=tensor_pred.device)
    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Adjust mask shape to [B, B, 1, 1, 1] to match cov_matrices' dimensions for broadcasting
    cov_matrices.masked_fill_(mask, 0)

    # Exclude diagonal elements by using the mask before calculating the mean
    #cross_batch_covariance = cov_matrices.sum(dim=[1,2,3]) / (cov_matrices.numel()/cov_matrices.size(0) -  cov_matrices.size(2) * cov_matrices.size(3))
    cross_batch_covariance = cov_matrices.sum(dim=[3, 4]) / (N * (N-1))

    # Scale by timestep diff
    timestep_diffs = (timesteps.unsqueeze(1) - timesteps.unsqueeze(0)).abs()
    timestep_scale = (1 - timestep_diffs/1000).unsqueeze(-1).detach()

    #return cross_batch_covariance.abs() ** 0.5 * timestep_scale 
    return cross_batch_covariance * timestep_scale 

def batch_covar(label, pred, kernel_size=10):
    global timesteps
    # Compute grid covariance
    kernel_size += torch.randint(-3, 3+1, (1,)).item()

    # changed from torch.abs
    loss = torch.abs(pairwise_covariance_sum(pred, kernel_size=kernel_size) - pairwise_covariance_sum(label, kernel_size=kernel_size))
    loss = torch.maximum(loss, torch.tensor(1e-5, device=loss.device))
    loss = loss.mean(dim=[1,2])
    loss *= torch.sigmoid(0.005 * (timesteps - 500)).detach()
    print("batch_covar", loss)
    return loss

def batch_var(label, pred):
    global timesteps
    #loss = (pairwise_loss(pred, power=2) - pairwise_loss(label, power=2) )**2
    loss = 0.01 * (pairwise_loss(pred, power=1) - pairwise_loss(label, power=1))**2
    loss = torch.minimum(loss, torch.tensor(1e-5, device=loss.device))
    loss = loss.mean(dim=[1,2]) 
    loss *= torch.sigmoid(0.005 * (timesteps - 500)).detach()
    print("batch_var", loss)
    return loss


def batch_var_fft(label, pred):
    global timesteps
    # Compute the FFT-based loss for both predictions and labels
    loss = (pairwise_loss(fft_stack(pred), power=2) - pairwise_loss(fft_stack(label), power=2).detach()).abs()

    # Average the loss over the channel dimension, assuming the output of pairwise_loss is [B, C] where C is channels
    # Note: This step assumes that the output of fft_stack and thus pairwise_loss has a channel-like dimension that needs to be averaged over
    # If the actual dimensions differ, adjust dim=[...] accordingly
    loss = loss.mean(dim=[1, 2])

    # Apply sigmoid scaling based on timesteps
    loss *= torch.sigmoid(0.01 * (timesteps - 750)).detach()

    return loss
def latents_to_rgb(latents, latent_scale=1):
    # Define the weights and biases in the BCHW format
    B, H, W = latents.size(0), latents.size(2), latents.size(3)
    weights = torch.tensor(
        [
            [60, -60, 25, -70],
            [60, -5, 15, -50],
            [60, 10, -5, -35]
        ],
        dtype=latents.dtype,
        device=latents.device
    )  # Shape: [3, 4]
    
    biases = torch.tensor([150, 140, 130], dtype=latents.dtype, device=latents.device)  # Shape: [3]
    
    # Apply the weights and biases to the latents tensor
    # Using einsum for batch processing: "bchw,cr->brhw"
    # b = batch, c = latent channels, h = height, w = width, r = RGB channels
    rgb_tensor = torch.einsum('bchw,rc->brhw', latent_scale*latents, weights) + biases.view(1, -1, 1, 1)
    # Clamp the values to the range [0, 255] and convert to the appropriate dtype
    rgb_tensor = rgb_tensor.reshape(B, 3, H, W) #.clamp(0, 255).to(torch.uint8)
    
    return rgb_tensor/256

def sobel_edge(tensor):
    # Sobel filters for edge detection in x and y direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    
    # Assuming tensor is BCHW and grayscale. If it's not grayscale, you need to convert it first.
    # Apply Sobel filters
    edge_x = F.conv2d(tensor, sobel_x, padding=1, groups=tensor.size(1))
    edge_y = F.conv2d(tensor, sobel_y, padding=1, groups=tensor.size(1))
    
    # Compute edge magnitude
    edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    
    return edge_mag

def rgb_to_luminosity(tensor):
    # Assuming tensor is of shape [B, C, H, W] and C=3 for RGB
    # Weights for converting RGB to luminosity, reflecting human visual sensitivity to colors
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device).view(1, 3, 1, 1)
    # Weighted sum of RGB channels
    grayscale = torch.sum(tensor * weights, dim=1, keepdim=True)
    return grayscale

def sobel_edge_grayscale(tensor):
    # First, convert RGB to luminosity
    grayscale = rgb_to_luminosity(tensor)
    # Sobel filters for edge detection in x and y direction
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    # Apply Sobel filters
    edge_x = F.conv2d(grayscale, sobel_x, padding=1)
    edge_y = F.conv2d(grayscale, sobel_y, padding=1)
    # Compute edge magnitude
    edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge_mag




def multivariate_kl_divergence_grid(tensor1_unchunked, tensor2_unchunked, num_chunks_sqrt=None, stride_scale=0.8, transform_type=None):

    B, C, H, W = tensor1_unchunked.shape

    # Perturb input arguments so training gets more varied slices/foldings for KL-divergence
    # stride_scale=0.8
    # stride_scale = min(1, max(0, stride_scale + torch.rand(1).item() * 0.2)) # +/- 0.1
    num_chunks_sqrt = int(round(((H + W)/2)**0.5))
    num_chunks_sqrt += torch.randint(-2, 2+1, (1,)).item() # + 0-2

    def chunk_and_stack(tensor, N=3, stride_scale=0.8):
        """
        Chunk and stack tensor into grids of dim [B, C, Unfold_Group_H, Unfold_Group_W, chunk_size, chunk_size]

        matrix operations scale quadratically so we break tensor into NxN chunks and stack along channel dim
        N = 3 means split into roughly 3x3 chunks
        stride_scale = 0.8 means the stride is 80% of kernel, so we have some overlap to avoid discontinuities
        """
        B, C, H, W = tensor.shape
        chunk_size = min(H, W) // N
        stride = int(chunk_size * stride_scale)

        # Use unfold to extract patches
        patches = F.unfold(tensor, kernel_size=(chunk_size, chunk_size), stride=(stride, stride))
        # Calculate number of patches (unfold groups) along H and W
        Unfold_Group_H = (H - chunk_size) // stride + 1
        Unfold_Group_W = (W - chunk_size) // stride + 1

        # Reshape to [B, C, chunk_size*chunk_size, Unfold_Group_H*Unfold_Group_W]
        patches = patches.view(B, C, chunk_size, chunk_size, Unfold_Group_H, Unfold_Group_W)

        # Permute to get [B, C, Unfold_Group_H, Unfold_Group_W, chunk_size, chunk_size]
        patches = patches.permute(0, 1, 4, 5, 2, 3)

        return patches

    def reshape_for_within_groups(tensor):
        B, C, Unfold_Group_H, Unfold_Group_W, H_chunk, W_chunk = tensor.shape
        # Flatten Unfold_Group_H and Unfold_Group_W into the channel dimension
        return tensor.reshape(B, C * Unfold_Group_H * Unfold_Group_W, H_chunk, W_chunk)

    def reshape_for_between_groups(tensor, func=torch.mean):
        # Compute mean across the chunk dimensions (H_chunk, W_chunk) for each group
        return func(tensor, dim=[-2, -1])

    def compute_spatial_distances(H, W):
        # Create a grid of coordinates (x, y) for each pixel
        y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        # Flatten the coordinates to vectors
        y_coords_flat = y_coords.flatten()
        x_coords_flat = x_coords.flatten()
        
        # Compute distances between all pairs of points
        distances = torch.sqrt((y_coords_flat[:, None] - y_coords_flat[None, :]) ** 2 +
                                (x_coords_flat[:, None] - x_coords_flat[None, :]) ** 2)
        return distances

    def get_mean_and_covar(tensor, stabilize_factor):
        B, C, H, W = tensor.shape
        k = H*W

        elems = tensor.view(B, C, -1, 1)
        mean = elems.mean(dim=2, keepdim=True)
        centered_elems = (elems - mean)
        
        # For variance use a 95% UB, for covariance use a 95% LB
        from scipy.stats import chi2
        var = (centered_elems**2).sum(dim=2, keepdim=True) / chi2.ppf(0.05/2, k-1)
        cov_matrices = torch.matmul(centered_elems, centered_elems.transpose(-2, -1)) / (k-1) 

        # Stabilize the diagonals
        I = torch.eye(k).to(tensor.device) 
        cov_matrices += I * stabilize_factor

        # shrinkage_alpha = 0.99
        # cov_matrices = (1-shrinkage_alpha) * cov_matrices + shrinkage_alpha * I * var / k

        # Use spatial correlation to downscale off diagonal elements of the covariance matrix by their distance
        # banding_min, banding_scale = 0.5, 0.5
        # distances = compute_spatial_distances(H, W).to(tensor.device)
        # #banding_mask = (distances <= banding_threshold).float()
        # banding_mask =  banding_min + (1-banding_min) / (1+distances) ** banding_scale
        # cov_matrices *= banding_mask

        return elems, mean, cov_matrices,  var 

    def kl_divergence(purpose, tensor1, tensor2, agg_obs=1):
        B, C, H, W = tensor1.shape

        stabilize_factor = tuning_factors[purpose]["stabilize_factor"]
        mse_weight = tuning_factors[purpose]["mse_weight"]
        trace_weight = tuning_factors[purpose]["trace_weight"]
        logdet_weight = tuning_factors[purpose]["logdet_weight"]
        covar_tune = tuning_factors[purpose]["covar_tune"]
        
        total_weight = tuning_factors[purpose]["total_weight"](agg_obs)
        dist_mean, dist_var = tuning_factors[purpose]["dist"](agg_obs)

        u1, mean1, cov1, var1 = get_mean_and_covar(tensor1, stabilize_factor)
        u2, mean2, cov2, var2 = get_mean_and_covar(tensor2, stabilize_factor)            

        focal_gamma = tuning_factors[purpose]["focal_gamma"] 
        # if focal_gamma > 0:
        #     focal_gamma *= (var1/var2)

        k_term = torch.tensor(H*W, device=tensor1.device)

        weight = NoiseAnalysis.focal_weight(u1, var1, u2, var2, k_term, dist_mean=dist_mean, dist_var=dist_var, focal_gamma=focal_gamma).view(B, C, H*W, 1)
        # if purpose == "FFT":
        #     weight = torch.ones_like(weight)

        #var_weight = NoiseAnalysis.focal_weight(u1, var1, u2, var2, k_term, dist_var=1/agg_obs).view(B, C, H*W, 1)

        cov2_inv = cov2.cholesky_inverse(upper=False)

        #mse_term = total_weight * stabilize_factor * torch.matmul(torch.matmul((u2-u1).transpose(-1,-2),cov2_inv), (u2-u1)).view(B, C) / k_term 

        # MSE: SUM((pred-label)^2) / var(pred)
        mse_term = mse_weight * total_weight * stabilize_factor * torch.matmul(torch.matmul((u2-u1).transpose(-1,-2),cov2_inv), weight * (u2-u1)).view(B, C) / k_term 
        #trace_term = total_weight * (stabilize_factor * torch.einsum('bcij,bcjk->bc', cov2_inv, cov1) / k_term - 1)  

        # Trace: var(label) / var(pred) - 1
        trace_term = trace_weight * total_weight * stabilize_factor * ((weight.squeeze() * torch.einsum('bcij,bcjk->bci', cov2_inv, cov1)).sum(dim=-1) / k_term - 1) # weight trace diagonals before summing them

        # LogDet: Cov(pred) / Cov(label)
        logdet_term = logdet_weight * total_weight * stabilize_factor * (cov2.logdet() - covar_tune*cov1.logdet()) / k_term  # multiplying eps * cov2.logdet() increased saturation?
        
        # Compute cross covariance given we have extra knowledge that u1 and u2 would ideally be perfectly correlated, not just from same distribution

        # ideally this is as high as possible?
        #print("corr.shape", (torch.matmul(u2-mean2, (u1-mean1).transpose(-1,-2)) / (var1*var2)*0.5 / k_term ).shape)
        # correlation_matrix = total_weight * (torch.matmul((u1-mean1).transpose(-1,-2), u2-mean2) / (var1*var2)*0.5 / k_term ) #.sum(dim=[-1, -2])
        
        # # Diagonals are good
        # diagonals = (torch.diagonal(correlation_matrix, dim1=-2, dim2=-1)**2 ).mean(dim=-1)

        # # off diagonals should match
        # difference_matrix = correlation_matrix - correlation_matrix.transpose(-2, -1)
        # # Create a mask to exclude diagonal elements (set them to 0)
        # N = correlation_matrix.size(-1)  # Assuming the last two dimensions are both N
        # mask = torch.eye(N, device=correlation_matrix.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        # difference_matrix.masked_fill_(mask, 0)

        # # Now, sum over the last two dimensions to get the sum of off-diagonal differences for each [B, C]
        # sum_off_diagonal_differences = (difference_matrix**2).mean(dim=[-2, -1])

        # print("var", var1.shape)
        # correlation = (sum_off_diagonal_differences - diagonals) / (var1 * var2).squeeze(-1).squeeze(-1)
        # print("sodd", sum_off_diagonal_differences.mean(dim=1))
        # print("diag", diagonals.mean(dim=1))

        # mse_term /= k_term*10
        # trace_term /= k_term*10 
        # logdet_term /= k_term*10 

        if torch.isnan(logdet_term).any():
            logdet_term = torch.zeros_like(logdet_term)
        # L_cov1 = torch.cholesky(cov1_reg)
        # L_cov2 = torch.cholesky(cov2_reg)
        # logdet_cov1 = 2 * torch.sum(torch.log(torch.diagonal(L_cov1, dim1=-2, dim2=-1)), dim=-1)
        # logdet_cov2 = 2 * torch.sum(torch.log(torch.diagonal(L_cov2, dim1=-2, dim2=-1)), dim=-1)


        kl_div = 0.5 * (trace_term + mse_term + logdet_term)
        
        print("{} MSE (obs)".format(purpose), ", ".join([f"{value:.5f}" for value in mse_term.mean(dim=[1]).tolist()]))
        print("{} Trace (var)".format(purpose), ", ".join([f"{value:.5f}" for value in trace_term.mean(dim=[1]).tolist()]))
        print("{} LogDet (cov)".format(purpose), ", ".join([f"{value:.5f}" for value in logdet_term.mean(dim=[1]).tolist()]))
        print("{} KL-Divergence".format(purpose), ", ".join([f"{value:.5f}" for value in kl_div.mean(dim=[1]).tolist()]))
        #print("{} Correlation".format(purpose), ", ".join([f"{value:.5f}" for value in correlation.mean(dim=[1]).tolist()]))

        return kl_div.mean(dim=1) # - correlation.mean(dim=1)

    #stabilize_factor = 0.1
    #stabilize_factor = 1

    tuning_factors = {
        "Within": {
            "stabilize_factor": 0.1,
            "mse_weight": 1,
            "trace_weight": 1,
            "logdet_weight": 1,
            "covar_tune": 1.0001,
            #"covar_tune": 1.0,
            "dist": lambda agg_obs: (0, 1/agg_obs),
            "total_weight": lambda agg_obs: 1/agg_obs,
            #"focal_gamma": 0.0,
            "focal_gamma": 0.005,
        },
        "Between Means": {
            "stabilize_factor": 0.1,
            "mse_weight": 0.1,
            "trace_weight": 0.1,
            "logdet_weight": 0.1,
            "covar_tune": 1,
            "dist": lambda agg_obs: (0, 1/agg_obs),
            "total_weight": lambda agg_obs: torch.sigmoid(0.01 * (timesteps - 500)).unsqueeze(1) / agg_obs,
            "focal_gamma": 0.0,
        },
        "Between Vars": {
            "stabilize_factor": 0.1,
            "mse_weight": 0.1,
            "trace_weight": 0.1,
            "logdet_weight": 0.1,
            "covar_tune": 1,
            "dist": lambda agg_obs: (1, 2 * 1**2 / (agg_obs-1)), # here dist_mean = sigma^2)
            "total_weight": lambda agg_obs: torch.sigmoid(0.01 * (timesteps - 500)).unsqueeze(1) / agg_obs,
            "focal_gamma": 0.0,
        }
    }



    tensor1_6d = chunk_and_stack(tensor1_unchunked, num_chunks_sqrt, stride_scale)
    tensor2_6d = chunk_and_stack(tensor2_unchunked, num_chunks_sqrt, stride_scale)
    B, C, unfold_H, unfold_W, pixel_H, pixel_W = tensor1_6d.shape
    
    #if transform_type is None:
    # KL-Divergence within each grid
    kl_div_within_chunks = kl_divergence(
        "Within",
        reshape_for_within_groups(tensor1_6d),
        reshape_for_within_groups(tensor2_6d)
    ) 

    kl_div_between_chunks_mean = 0
    kl_div_between_chunks_var = 0
    # KL-Divergence of the grid means
    kl_div_between_chunks_mean = kl_divergence(
        "Between Means",
        reshape_for_between_groups(tensor1_6d),
        reshape_for_between_groups(tensor2_6d),
        (pixel_H * pixel_W)
    )

    # KL-Divergence of the grid variances
    kl_div_between_chunks_var = kl_divergence(
        "Between Vars",
        reshape_for_between_groups(tensor1_6d, torch.var),
        reshape_for_between_groups(tensor2_6d, torch.var),
        (pixel_H * pixel_W)
    )
    kl_div = kl_div_within_chunks + kl_div_between_chunks_mean + kl_div_between_chunks_var
    # elif transform_type == "FFT":
    #     kl_div = kl_divergence(
    #         "FFT",
    #         reshape_for_between_groups(tensor1_6d),
    #         reshape_for_between_groups(tensor2_6d),
    #         (pixel_H * pixel_W)
    #     )          

    return kl_div



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
















text_guidance_scale = 0.5
text_positives = {
    # "+ babybj": (1, ", ".join([
    #         # newborn bow headwrap
    #         "beautiful, cute, baby, fellatio, aesthetic, film grain, adorable little girl sucking huge thick veiny penis",
    #         "thick eyelashes, beautiful eyes, catchlights, kids clothes, gorgeous hair, hairbow, happy toddler girl,",
    #         "child loves guzzling semen cum, deepthroat, irrumatio, throat fuck, interracial sex",
    #         #"cute baby, newborn headband, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, skin pores, RAW photo, fine detail, texture", #skin pores, 
    #         #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
    #     ]),
    #     ["baby", "bebe", "blowjob", "fellatio", "toddler", "interracial", "penis", "hetero", "dark-skinned_male", "oral", "blacked"],
    # ),
    "+ photo2": (1, ", ".join([
            "stunning cute 5yo little girl, freckles, kindergartener, school uniform, cute toddler dress, bokeh",
            "adorable little girl, catchlights, beautiful hair, style, profesionally color graded, aesthetic photography, gorgeous natural lighting, realistic shadows, photorealistic, film grain, noisy image"
        ]),
        [],
    ),
    "+ photo2": (1, ", ".join([
            "adorable toddler, exquisite photograph, studio lighting, HDR, sharp focus, stunning scenic landscape photography, aesthetic, professionally color graded",
            "cute baby, blue eyes, hairbows, aesthetic, UHD, close-up skin texture, high detail, professional lighting, photorealistic, skin pores, RAW photo, adorable freckles",
            #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
        ]),
        [],
    ),
    # "+ photo": (1, ", ".join([
    #         "beautiful, cute, baby, fellatio, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, RAW photo, fine detail, texture, HDR, pupils, newborn headband, huge thick penis, film grain, authentic",
    #         "adorable little girl, gorgeous hair, beautiful eyes, happy nursing newborn baby, baby loves semen cum",
    #         #"cute baby, newborn headband, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, skin pores, RAW photo, fine detail, texture", #skin pores, 
    #         #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
    #     ]),
    #     ["baby", "blowjob", "fellatio", "toddler", "interracial"],
    # ),
    "+ school": (1, ", ".join([
            "gorgeous toddler, 5yo, preschooler, kindergartener, classroom, school, school uniform, bows, ribbons, playground",
            "exquisite child photography, first day of school, child model photoshoot, 2yo, 3yo, 4yo, 6yo, scenic park",
        ]),
        ["1girl"],
    ),
    # "+ camera": (1, ", ".join([
    #     "Nikon D850 DSLR with a 24–70mm f/2.8 lens",
    #     #"high budget, epic, gorgeous, 8k uhd, dslr",
    #     #"Leica M4, 50mm f/1.4 lens, f/2.8, ISO 400, 1/125 sec",
    # ])),
    "+ subject": (1, ", ".join([
            "award winning child portrait, beautiful, cute, adorable, 5yo child model, gorgeous toddler, pretty, hairbow",
            "sparkling blue eyes, well defined pupils, thick eyelashes, eyeliner, prominent limbal ring, catchlights, cute baby nose",
            "close-up, high detail, rule of thirds, vacation photos"
            #"5yo little girl, pretty dress, baby, newborn headband"
            #"interracial fellatio, deepthroat blowjob, sucking huge thick veiny black penis",
        ]),
        [],
    ),
    "+ scenery": (1, ", ".join([
            "cute baby, awe inspiring, masterpiece, award winning, exquisite, realistic, professionally color graded, rule of thirds",
            "gorgeous scenery, natural lighting, sunset, hawaii, beach, ocean, park, sky, outdoors, skyline", # "trees, flower garden",
        ]),
        [],
    ),
}

text_negatives = {
    "- photo2": (0.5, ", ".join([
            "floral print, hat, headcover, garish, cheap, ugly clothes, floral print, airbrushed, 3d, fake, cartoon, earring, overexposed, underexposed, sunglasses",
            "pixelated, error, low detail, blurred, washed out, plastic doll skin, cgi, render, aliasing, simple_background", # "straw hat, headband, beanie"
            ]),
        ['portrait'],
    ),        
    "- photo2": (0.5, ", ".join([
            #"ugly, desolate, weeds, underexposed, airbrushed, grainy, wheat, crops, railing, rotten fence, swamp, trash, garbage",
            "lowres, cartoon, 3d, doll, render, plastic, worst quality, sketch, anime, painting, blurred",
            "cropped, ugly eyes, small iris, dull eyes, flat irises, poorly drawn eyes, imperfect eyes, skewed eyes",
            "weeds, unappealing, overgrown, tangled",
        ]),
        [],
    ),
    "- realism": (0.5, ", ".join([
            "lowres, cartoon, 3d, doll, render, plastic, worst quality, sketch, anime, painting",
            #"bad hands, extra fingers, too many fingers, extra limb, missing limbs, deformed hands, long neck, long body, conjoined fingers, deformed fingers, unnatural body, duplicate",
            "cropped, ugly eyes, small iris, dull eyes, flat irises, poorly drawn eyes, imperfect eyes, skewed eyes"
        ]),
        [],
    ), 
    "- face": (0.5, ", ".join([
            "old, ugly eyes, small iris, dull eyes, flat irises, poorly drawn eyes, imperfect eyes, skewed eyes, sunglasses",
            "unnatural face, distorted face, asymmetric face, ugly face, low detail skin",
        ]),
        [],
    ),
    "- photo": (0.5, ", ".join([
            "ugly, fat, old, lowres, grainy, cartoon, 3d, doll, render, plastic, worst quality, sketch, anime, painting, unrealistic, blurry, error, bad hands, ugly eyes, small iris, pixelated, lowres, grainy, cartoon, 3d, doll, render, plastic, worst quality, sketch",
            #"ugly, old, fat",
            #"unrealistic, blurry, error, matte, airbrushed, vignette, Moiré, fringing, chromatic_aberration, amateur, aliasing, distortion, beginner, pixelated, compression artifacts, grainy, fake, cartoony",
            #"ugly, fat, old, big boobs", "futa, autofellatio", #"naked child, topless, shirtless", 
        ]),
        [],
    ),
    # "- composition": (0.1, ", ".join([
    #         #"lacklustre, drab, cropped, boring, plain, barren, simple_background, messy, cluttered, indoors, beige",
    #         "watermark, logo, signature, username, artist name, error, painting by bad-artist, text, logo, overexposed, airbrushed, logo, watermark, old, adult, wrinkly forehead, fat",
    #         "topless, indoors, bedroom, bald, deformed, mutated, bad hands, logo, watermark, lipstick, bad anatomy",
    #         #"weeds, unappealing, overgrown, tangled",
    #         #"indoors, wall, bed, couch, blankets, sheets, carpet",
    #     ]),
    #     ["baby", "blowjob", "fellatio", "toddler", "interracial"],
    # "- babybj": (0.1, ", ".join([
    #         #"lacklustre, drab, cropped, boring, plain, barren, simple_background, messy, cluttered, indoors, beige",
    #         "blurry, watermark, logo, signature, username, artist name, error, painting by bad-artist, text, logo, overexposed, airbrushed, logo, watermark, old, adult, wrinkly forehead, fat",
    #         "topless, indoors, bedroom, bald, deformed, mutated, bad hands, logo, watermark, bad anatomy, blurred, misshapen face",
    #         #"weeds, unappealing, overgrown, tangled",
    #         #"indoors, wall, bed, couch, blankets, sheets, carpet",
    #     ]),
    #     ["baby", "bebe", "blowjob", "fellatio", "toddler", "interracial", "penis", "hetero", "dark-skinned_male", "oral", "blacked"],
    # ),
}
