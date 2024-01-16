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
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from library import sdxl_model_util
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision
from typing import Tuple, List
import numpy as np
import pandas as pd
from diffusers import AutoencoderTiny

# for network_module in network.text_encoder_loras:
#     random_dropout = random.uniform(0, args.network_dropout)
#     network_module.dropout = random_dropout


    
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
        clip_model, _ = self.manager.vit_model
        text_input = clip.tokenize(prompt).to(self.device)
        with torch.no_grad():
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

#from skimage.feature import greycomatrix, greycoprops
#from skimage.feature import local_binary_pattern
class NoiseAnalysis:
            
    @staticmethod
    def noise(pil_images):
        """
        Computes the noise score for a batch of PIL images.

        Parameters:
        pil_images (List[Image.Image]): List of PIL images.

        Returns:
        torch.Tensor: Tensor of noise scores for each image in the batch.
        """
        noise_scores = []

        with torch.no_grad():
            for img in pil_images:
                # Convert PIL image to grayscale and to a numpy array
                gray_img = np.array(img.convert("L"))

                # Calculate the standard deviation (noise)
                noise_score = np.std(gray_img) / 64

                noise_scores.append(noise_score)

        return torch.tensor(noise_scores, device=accelerator.device) 


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

        def transform_skewness(skewness_tensor):
            # Take the sign of each skewness value
            signs = skewness_tensor.sign()

            # Apply the cubic root transformation to the absolute values
            transformed_skewness = torch.abs(skewness_tensor) ** (1/3)

            # Reapply the signs to preserve the direction of skewness
            transformed_skewness *= signs

            return transformed_skewness

        return transform_skewness(skewnesses_tensor)

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
        with torch.no_grad():
            for i in range(tensor.shape[0]):  # Iterate over each item in the batch
                # Compute the standard deviation of the flattened tensor
                std_dev = tensor[i].view(-1).std().cpu().item()
                
                # Calculate the differential entropy for a normal distribution
                entropy = 0.5 * np.log2(2 * np.pi * np.e * std_dev**2)
                entropies.append(entropy)
            return torch.tensor(entropies, device=tensor.device)

    @staticmethod
    def sharpness_tensor(images_tensor):
        # Check if the tensor is already 4D: [batch_size, channels, height, width]
        if images_tensor.ndim != 4:
            raise ValueError("Input tensor must be 4-dimensional [batch_size, channels, height, width]")

        # Convert images to grayscale if they have 3 channels (RGB)
        if images_tensor.shape[1] == 3:
            # Simple averaging over the RGB channels to convert to grayscale
            images_tensor = images_tensor.mean(dim=1, keepdim=True)
        
        # # Normalize images to [0, 1] range
        # images_tensor = images_tensor.to(torch.float32) / 255.0

        # Define a Laplacian kernel
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]], dtype=images_tensor.dtype, device=images_tensor.device)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

        # Apply the Laplacian filter
        laplacian = torch.nn.functional.conv2d(images_tensor, laplacian_kernel, padding=1)

        # Compute the standard deviation for each image
        sharpness_values = laplacian.view(laplacian.shape[0], -1).std(dim=1) * 2

        return sharpness_values

    @staticmethod
    def sharpness_PIL(pil_images):
        sharpness_scores = []

        for img in pil_images:
            # Convert PIL image to grayscale
            gray_img = img.convert('L')

            # Convert PIL image to NumPy array
            np_img = np.array(gray_img)

            # Apply Laplacian operator
            laplacian = cv2.Laplacian(np_img, cv2.CV_64F)

            # Compute the standard deviation (sharpness score)
            sharpness = laplacian.std()
            sharpness_scores.append(sharpness)

        # Convert list of sharpness scores to a tensor
        sharpness_tensor = torch.tensor(sharpness_scores, dtype=torch.float32)

        return sharpness_tensor

class DiscriminatorManager:
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

    def apply_discriminator_losses(self, base_loss, timesteps, original_latents, noise, noisy_latents, noise_pred, step, output_name, output_dir):
        
        original = {}
        denoised = {}
        
        original["noise"] = noise
        denoised["noise"] = noise_pred

        original["latent"] = original_latents
        denoised["latent"] = self.remove_noise(noisy_latents, noise_pred, timesteps)

        # Decode latents and get images, embeddings
        self.vae_model.to(self.device)
        self._process_batch(original, denoised, step, timesteps, output_name, output_dir)

        # Apply each discriminator
        modified_loss = 0
        all_diagnostics = []
        for discriminator_name, discriminator in self.discriminators.items():
            original_scores = discriminator.compute_scores(original[discriminator.input_type])
            denoised_scores = discriminator.compute_scores(denoised[discriminator.input_type])

            discriminator_loss = discriminator.loss_func(original_scores, denoised_scores)
            
            """
            Q = alphas_cumprod_t
            SNR = Q / (1-Q)
            debias = 1 / SNR^0.5
            debias = sqrt((1-Q)/Q)
            """
            if discriminator.input_type != 'noise':
                discriminator_loss = self.scale_losses(discriminator_loss, timesteps)
            
            discriminator_loss *= discriminator.weight

            # Try to normalize loss scale across different timesteps
            #discriminator_loss = scale_v_prediction_loss_like_noise_prediction(discriminator_loss, timesteps, noise_scheduler)

            #self._print_diagnostics(discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss)

            diagnostics = self._accumulate_diagnostics(discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss)
            all_diagnostics.extend(diagnostics)

            modified_loss += discriminator_loss

        # Convert accumulated data to DataFrame
        df = pd.DataFrame(all_diagnostics)

        # Pivoting and displaying the data
        if self.print_diagnostics:
            self._pivot_and_display(df)

        return base_loss + modified_loss

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

    # def _pivot_and_display(self, df, row_order=['Type', 'Batch'], max_row_length=120):
    #     #df = df.drop_duplicates(subset=row_order + ['Discriminator'])

    #     # Pivot the DataFrame
    #     pivoted_df = df.pivot_table(index=row_order, columns='Discriminator', values='Value', aggfunc='first')

    #     # Sort columns based on the order in self.discriminators
    #     discriminator_order = list(self.discriminators.keys())  # Assuming this is the correct order
    #     pivoted_df = pivoted_df[discriminator_order + ['latent'] if 'latent' in pivoted_df.columns else discriminator_order]

    #     # Compute maximum column header length
    #     num_columns = len(pivoted_df.columns) + 1  # +1 for 'Batch'
    #     space_for_columns = max_row_length - len('Type B  ')
    #     max_col_length = max(7, space_for_columns // num_columns)

    #     # Abbreviate discriminator names based on computed length
    #     pivoted_df.columns = [col[:max_col_length].strip() for col in pivoted_df.columns]

    #     # Round the data to desired decimal places
    #     rounded_df = pivoted_df.round(2)

    #     # Add 'Total' column by summing across the specified columns, only for 'Loss'
    #     # Filter DataFrame for 'Loss' rows and compute total
    #     loss_rows = rounded_df.xs('Loss', level='Type')
    #     total_loss = loss_rows.sum(axis=1)

    #     # Create a 'Total' column with the same multi-level index as rounded_df
    #     total_column = pd.Series('', index=rounded_df.index)
    #     total_column.loc[('Loss', slice(None))] = total_loss.values
    #     rounded_df['total'] = total_column

    #     # Reorder 'Type' values
    #     type_order = ['Loss', 'Orig', 'Deno']
    #     rounded_df = rounded_df.reindex(type_order, level='Type')

    #     # Replace NaN with blank spaces for 'Orig' and 'Deno' and for latent column
    #     rounded_df = rounded_df.fillna('')

    #     # Modify the index to collapse headers
    #     rounded_df.index = [' '.join(map(str, idx)) for idx in rounded_df.index]

    #     # Convert DataFrame to string and print
    #     df_string = rounded_df.to_string(index=True, justify='right')
    #     tqdm.write(df_string)

    # Inside DiscriminatorManager
    def _process_batch(self, original, denoised, step, timesteps, output_name, output_dir):
        original["decode"], denoised["decode"] = self._decode_latents(original["latent"], denoised["latent"])
        original["image"], denoised["image"] = self._get_PIL_images(original["decode"], denoised["decode"])
        self._save_image_pairs(original["image"], denoised["image"], step, timesteps, output_name, output_dir)
        original["embedding"], denoised["embedding"] = self._get_image_embeddings(original["image"], denoised["image"])
        original["decode"].cpu()
        denoised["decode"].cpu()
        

    def _decode_latents(self, 
                        original_latents: torch.Tensor, 
                        denoised_latents: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes latent tensors into image tensors.

        Parameters:
        original_latents (torch.Tensor): The original latent tensors.
        denoised_latents (torch.Tensor): The denoised latent tensors.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors representing the decoded original and denoised images.
        """
        def decode_latents_single(latents):
            with torch.no_grad():
                decodes = []
                for i in range(latents.size(0)):
                    latent = latents[i].unsqueeze(0).to(self.vae_model.dtype)
                    decode = self.vae_model.decode(latent).sample
                    decodes.append(decode)

                return torch.cat(decodes, dim=0)
        return decode_latents_single(original_latents), decode_latents_single(denoised_latents)

    def _get_PIL_images(self, 
                        original_decodes: torch.Tensor, 
                        denoised_decodes: torch.Tensor
                        ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Converts decoded images to a list of PIL images.

        Parameters:
        original_decodes (torch.Tensor): Decoded original images.
        denoised_decodes (torch.Tensor): Decoded denoised images.

        Returns:
        Tuple[List[Image.Image], List[Image.Image]]: Lists of PIL images for original and denoised images.
        """
        def to_PIL(images):
            with torch.no_grad():
                # Rescale from [-1, 1] to [0, 1]
                images = (images / 2 + 0.5).clamp(0, 1)
                # Convert to 0-255 range and change layout to [batch, height, width, channels]
                images = images.cpu().permute(0, 2, 3, 1).float().numpy()
                images = np.nan_to_num(images, nan=0.0, posinf=0.0, neginf=0.0)
                images = (images * 255).round().astype("uint8")
                return [Image.fromarray(im) for im in images]

        return to_PIL(original_decodes), to_PIL(denoised_decodes)



    def _get_image_embeddings(self, 
                                original_images: torch.Tensor, 
                                denoised_images: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates embeddings for images using a CLIP model.

        Parameters:
        original_images (Tuple[List[Image.Image]): List of original images in PIL format.
        denoised_images (Tuple[List[Image.Image]): List of denoised images in PIL format.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Embeddings for original and denoised images.
        """
        clip_model, preprocess = self.vit_model

        def get_embeddings(images):
            with torch.no_grad():
                preprocessed_images = [preprocess(img).unsqueeze(0) for img in images]
                preproc_images_batch = torch.cat(preprocessed_images, dim=0).to(self.device)
                embeddings = clip_model.encode_image(preproc_images_batch)
                return embeddings / embeddings.norm(dim=-1, keepdim=True)

        return get_embeddings(original_images), get_embeddings(denoised_images)


    def _save_image_pairs(self, original_images, denoised_images, step, timesteps, output_name, output_dir):

        def _save_image_pair(index, original_pil, denoised_pil, save_dir, step, timesteps):
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
            thread = Thread(target=_save_image_pair, args=(i, original_images[i], denoised_images[i], save_dir, step, timesteps))
            thread.daemon = True  # Set thread as daemon
            thread.start()


    # def _print_diagnostics(self, discriminator_name, timesteps, original_scores, denoised_scores, discriminator_loss, base_loss):
    #     with torch.no_grad():
    #         if denoised_scores.ndim == 0:
    #             print(f"\n{discriminator_name} Scores and Losses:")
    #             print(f"{'Index':<6} {'Original Score':<15} {'Denoised Score':<15} {'Discriminator Loss':<20} {'Latent Loss':<15}")
    #             print(f"{'All':<6} {original_scores.item():<15.2f} {denoised_scores.item():<15.2f} {discriminator_loss.item():<20.2f} {base_loss.item():<15.2f}")
    #         else:
    #             original_scores_np = original_scores.cpu().numpy()
    #             denoised_scores_np = denoised_scores.cpu().numpy()
    #             discriminator_loss_np = discriminator_loss.cpu().numpy()
    #             base_loss_np = base_loss.cpu().numpy()

    #             print(f"\n{discriminator_name} Scores and Losses:")
    #             for i in range(len(denoised_scores_np)):
    #                 print(f"{i:<6} {original_scores_np[i]:<15.2f} {denoised_scores_np[i]:<15.2f} {discriminator_loss_np[i]:<20.2f} {base_loss_np[i]:<15.2f}")






















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




















