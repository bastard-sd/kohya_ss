import importlib
import argparse
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
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from torch.nn.parallel import DistributedDataParallel as DDP

from accelerate.utils import set_seed
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
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
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import sys
from sdxl_gen_img import get_weighted_text_embeddings as get_weighted_sdxl_text_embeddings
from discriminator import DiscriminatorManager
import discriminator

class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self, args: argparse.Namespace, current_loss, avr_loss, lr_scheduler, keys_scaled=None, mean_norm=None, maximum_norm=None
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()

        if args.network_train_text_encoder_only or len(lrs) <= 2:  # not block lr (or single block)
            if args.network_train_unet_only:
                logs["lr/unet"] = float(lrs[0])
            elif args.network_train_text_encoder_only:
                logs["lr/textencoder"] = float(lrs[0])
            else:
                logs["lr/textencoder"] = float(lrs[0])
                logs["lr/unet"] = float(lrs[-1])  # may be same to textencoder

            if (
                args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                )
        else:
            idx = 0
            if not args.network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        pass

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        setup_logging(args, reset=True)

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # tokenizerは単体またはリスト、tokenizersは必ずリスト：既存のコードとの互換性のため
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        noise_scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        args.zero_terminal_snr = True

        print(f"vae_scale_factor: {2 ** (len(vae.config.block_out_channels) - 1)}")
        #vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float32)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        self.cache_text_encoder_outputs_if_needed(
            args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
        )

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        network_has_multiplier = hasattr(network, "set_multiplier")

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # 後方互換性を確保するよ
        try:
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
        except TypeError:
            accelerator.print(
                "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
            )
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # dataloaderを準備する
        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training.")
            unet_weight_dtype = torch.float8_e4m3fn
            te_weight_dtype = torch.float8_e4m3fn

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)
                # nn.Embedding not support FP8
                t_enc.text_model.embeddings.to(dtype=(weight_dtype if te_weight_dtype != weight_dtype else te_weight_dtype))

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if train_unet:
            unet = accelerator.prepare(unet)
        else:
            unet.to(accelerator.device, dtype=unet_weight_dtype)  # move to device because unet is not prepared by accelerator
        if train_text_encoder:
            if len(text_encoders) > 1:
                text_encoder = text_encoders = [accelerator.prepare(t_enc) for t_enc in text_encoders]
            else:
                text_encoder = accelerator.prepare(text_encoder)
                text_encoders = [text_encoder]
        else:
            pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

        network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(network, optimizer, train_dataloader, lr_scheduler)

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
        }

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
        global_step = 0

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs
            )

        loss_recorder = train_util.LossRecorder()
        del train_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)
                
                
                
                
                
                
                
                
                
                
        if args.discriminator_config_path:
            discriminator_config_json = args.discriminator_config_path
        else:
            discriminator_config_json = os.path.join(args.train_data_dir, 'discriminator_config.json')
            
        discriminator_manager = DiscriminatorManager(discriminator_config_json, accelerator.device, noise_scheduler, tokenizers, text_encoders, self.is_sdxl, save_image_steps=1, print_diagnostics=True, accelerate=accelerator, args=args)
        vae = discriminator_manager.vae_model









        quality = [
            "aesthetic", "exquisite", "stunning", "breathtaking", "award-winning", "majestic", "unparalleled", "world-class",
            "beautiful", "photogenic", "innovative", "visionary", "masterful", "legendary", "pioneering", "spectacular",
            "sharp", "vibrant", "crisp", "detailed", "meticulous", "polished", "sophisticated", "refined", "iconic", "picturesque",
            "pristine", "elegant", "sleek", "dynamic", "vivid", "luminous", "radiant", "harmonious", "balanced",  "opulent", 
            "timeless", "immortal", "peerless", "sublime", "revolutionary", "photorealistic", "cinematic", "masterpiece", "master-class",
        ]

        emotional = [
            "tranquil", "nostalgic", "atmospheric", "sumptuous", "serene", "emotionally moving", "idyllic",
            "spellbinding", "ethereal", "poetic narrative", "expressive", "dramatic", "inviting", "inspiring",
            "inspirational", "unique", "original", "evocative", "thrilling", "passionate", "mood", "ambiance",
            "intimate", "soul-stirring", "heartwarming", "melancholic", "whimsical", "dreamy", "haunting",
            "provocative", "sensual", "sentimental", "vibrant emotion", "touching", "rousing", "mysterious",
            "enigmatic", "alluring", "bewitching", "charming", "lyrical", "magical", "otherworldly", "surreal",
            "gripping", "compelling", "engrossing", "captivating", "mesmerizing", "enchanting", "ambient", "mystical",
        ]

        lighting = [
            "natural lighting", "studio lighting", "sunset", "golden hour", "rim lighting", "filtered light",
            "volumetric lighting", "backlighting", "twilight", "blue hour", "magic hour", "diffused lighting",
            "soft shadows", "dappled light", "neon glow", "silhouette lighting", "mood lighting",
            "key lighting", "fill lighting", "high-key lighting", "low-key lighting", "catch lights",
            "butterfly lighting", "rembrandt lighting", "loop lighting", "broad lighting", "short lighting",
            "split lighting", "kicker light", "hair light", "accent lighting", "cross lighting",
            "bounce lighting", "side lighting", "soft light", "ambient lighting", "sculptural lighting",
            "directional lighting", "spot lighting", "clamshell lighting", "beauty lighting",
            "light painting", "continuous lighting", "LED panels", "softboxes", "umbrellas", "octaboxes", "ring flash",
        ]

        attractive = [
            "cute", "adorable", "precious", "beautiful", "sweet", "glamorous", "angelic", "elegant", "radiant", 
            "charming", "whimsical", "enigmatic", "alluring", "lovely", "stunning", "gorgeous", "bewitching", 
            "enticing", "fetching", "graceful", "dazzling", "exquisite", "magnetic", "divine", "splendid", 
            "winsome", "serene", "luminous", "vivacious", "lifelike", "fairylike", "angelic",
            "dreamy", "svelte", "refined", "polished", "sophisticated", "sublime", "delightful", "enchanting", 
            "irresistible", "impeccable", "breathtaking", "spellbinding",  "quaint", "chic", 
            "sleek", "ravishing", "appealing", "enticing", "lush", "peppy", "spirited", "zesty", "vibrant"
        ]
        subjects = [f + " girl" for f in [
            "cute little", "preschooler", "toddler", "kindergartener", "baby", "child model", "2yo", "3yo", "4yo", "5yo", "6yo", 
        ]]
        appearance = [
            "blonde", "ginger", "redhead", "albino", "brunette",
            "blushing cheeks", "nose freckles", "colorful hairbows", "creative hairstyles",
            "braided hair", "sparkling eyes", "joyful giggles", "curious gaze",
            "playful stance", "innocent charm", "sun-kissed skin", "wind-blown hair",
            "pigtails", "rosy cheeks", "wide-eyed", "cascading curls", "sparkling blue eyes",
            "mischievous grin", "twinkling laughter", "enchanting dimples",
            "floral wreaths", "bubble lips", "animated gestures", 
            "genuine delight", "pastel nails", "meticulous braids", "expressive eyes",
            "contemplative", "skipping steps", "graceful movement", "melodious voice",
            "warm laughter", "adventurous spirit", "shimmering ribbons", "glitter accents",
            "soft smiles", "storybook whimsy", "playful socks", "dewy glow",
            "tiptoe twirls", "ribbon twirls", "thick eyelashes", "beautiful eyes",
            "prominent limbal ring", "detailed iris texture", "perfect eyes", "skin pores", 
            # "dreamy look", 
        ]

        clothing = [
            "newborn headwrap",
            "tulle skirts", "polka dot dresses", "floral print frocks", "lace-trimmed leggings",
            "ballet tutus", "ribbon-tied headbands", "peter pan collars", "sequin embellished tops",
            "ruffled rompers", "denim overalls", "pastel cardigans", "gingham sun dresses",
            "striped t-shirts", "knitted sweaters", "pom-pom caps", "faux fur vests",
            "butterfly wings accessories", "sparkly flats", "frilly socks", "beaded bracelets",
            "embroidered jean jackets", "rainbow scarves", "unicorn prints", "star-patterned leggings",
            "cat-ear headbands", "sun hats with wide brims", "tiered peasant skirts", "vintage pinafores",
            "eyelet lace blouses", "lightweight chiffon tops", "woodland animal motifs", "princess-themed costumes",
            "sailor-style dresses", "velvet bows", "soft cotton pajamas", "dainty floral head wreaths",
            "toddler tutu dresses", "smocked sundresses", "eyelet sundresses", "soft jersey playsuits",
            "floral smocked dresses", "ruffled tank dresses", "gingham check dresses", "liberty print frocks",
            "polka dot bloomers", "frill-sleeve tops", "embroidered dungarees", "seersucker shorts",
            "knit cardigan sweaters", "velvet ribbon headbands", "mini denim skirts", "peplum tops",
            "crochet trim leggings", "bow-detail sandals", "lace collar tees", "chambray shirt dresses",
            "scallop edge shorts", "pleated skirtalls", "flounce sleeve blouses", "daisy print leggings",
            "patchwork fabric hats", "ruffle butt leggings", "linen blend jumpsuits", "corduroy overalls",
            "mary jane shoes", "ballet slipper socks", "lemon print sets", "ditsy floral rompers", "patterned leggings", 
            "watermelon slice purses", "rainbow appliqué cardigans", "bear ear beanies", "faux shearling coats",
            "plush velvet jumpsuits", "fringed moccasins", "ladybug raincoats", "sunflower headbands",
            "glittery tights", "heart-shaped sunglasses", "neon swimwear", "toddler bikini", "airy fairy costumes",
            "color-blocked raincoats", "beach cover-ups with tassels", "character-themed backpacks", "whimsical tutu",
            "crochet lace dresses", "polka dot rain boots", "tiered ruffle dresses", "lace dresses", "ballet flats",
        ]

        locations = [
            "ocean beach", "tropical paradise", "lush botanical garden", "cyberpunk city", "bustling metropolis", "city streets", "lush rainforest", "stonehenge",
            "Amalfi Coast, Italy", "Bora Bora", "coastal cliffs", "cobblestone streets", "waterfalls", "alpine meadows", "rooftop terrace", "tropical beaches", 
            "breathtaking vineyard", "mountain ranges", "national parks", "wonder of the world", "urban skyline", "rustic English village", "mirror-like lakes", 
            "ancient forest", "snow-capped village", "hidden valleys", "magical elven faerie forest", "grand palaces", "secluded beaches", "urban rooftops at night", "flower fields",
            "glacier-carved valleys", "volcanic landscapes", "otherworldly desert formations", "underwater worlds", "haunting ruins", "sky-high mountain peaks", "emerald green rice terraces" ,
            "fantasy-like castles", "luminous cityscapes at night", "isolated islands", "rose garden", "yosemite national park", "yellowstone national park",
            "bioluminescent bays", "geothermal springs", "ice formations in polar regions", "lavender farms at sunset", "dramatic stormy seas", "ancient monasteries in remote locations",
            "Giverny, France", "The Roman Colosseum, Italy", "The Canals of Venice, Italy", "Santorini, Greece", "The Blue Ridge Mountains, USA",  "The Sossusvlei Dunes, Namibia",
            "The Great Pyramids of Giza, Egypt", "Machu Picchu, Peru", "The Grand Canyon, USA", "Mount Fuji, Japan", "The Great Barrier Reef, Australia", 
            "The Lavender Fields of Provence, France", "The Serengeti Plains, Tanzania", "The Northern Lights, Various Arctic Locations", "The Cinque Terre, Italy", 
            "The Fjords of Norway", "The Acropolis, Greece", "The Sahara Desert", "The Scottish Highlands", "Banff National Park, Canada", "Angel Falls, Venezuela", 
            "The Floating Markets of Bangkok, Thailand", "The Vineyards of Tuscany, Italy","The Himalayas", "The Blue City of Chefchaouen, Morocco",
            "Cherry Blossom Groves in Kyoto, Japan", "The Palaces of Jaipur, India", "Monet's Garden in Giverny, France", "Keukenhof Gardens, Netherlands",
            "The Cliffs of Moher, Ireland", "The Old Town of Dubrovnik, Croatia", "The Dolomites, Italy", "Valley of Flowers, India", "Plitvice Lakes National Park, Croatia",
            "The Wisteria Tunnel at Kawachi Fuji Gardens, Japan", 
        ]

        composition = [
            "Golden Ratio in Portraiture for Natural Elegance", "Diagonal Method in Dynamic Composition",
            "Figure to Ground Relationship in Portraiture to Emphasize Subjects", "Color Blocking for Visual Impact in Fashion Photography",
            "Negative Space to Enhance Subject Isolation", "Leading Lines and Visual Pathways to Guide the Viewer’s Eye",
            "Texture and Pattern as Primary Subjects in Close-ups", "Frame Within a Frame for Depth and Focus",
            "Visual Weight and Balance in Group Portraits", "Symmetry and Asymmetry in Portrait Composition",
            "Chromatic Harmony and Discord in Color Grading for Emotional Impact", "Mood and Atmosphere through Weather Elements in Outdoor Shoots",
            "Narrative Imagery and Storytelling in Editorial Photography", "Psychological Impact of Color Usage in Expressive Portraits",
            "Artistic Use of Shadows and Highlights to Sculpt Faces", "Poetic Imagery in Visual Storytelling for Personal Projects",
            "Ethereal Light and Dreamy Atmospheres for Soft Portraits", "Conceptual Photography with Deep Messages",
            "Expressionist Techniques for Emotional Depth in Character Studies", "Art Nouveau Inspired Photography for Flowing Lines and Organic Forms",
            "Visual Puns and Clever Composition in Conceptual Art", "Optical Illusions in Photography for Engaging Portraits",
            "Minimalist Aesthetics for Stronger Subject Focus", "Juxtaposition in Street and Documentary Photography to Tell Human Stories",
        ]

        techniques = [
            "Focus Stacking for Hyperreal Detail", "Dodge and Burn for Dynamic Range",
            "Color Theory Application in Post-Processing", "Underwater Portraiture",
            "High-Speed Flash Freeze to Capture Emotion", "Digital Double Exposure for Conceptual Portraits",
            "Selective Color Isolation to Highlight Subjects", "Cinematic Videography Stills for Storytelling",
            "Environmental Portraiture with Natural Elements", "Fine Art Nudes in Nature",
            "Experimental Light Painting Techniques for Dynamic Imagery", "Shadow Play in Composition for Mood",
            "Mirror and Reflection Symmetry for Artistic Portraits", "Cinematic Lighting Techniques for Atmospheric Shots",
            "Renaissance Lighting in Portraits for Classical Beauty", "Baroque Influence in Dramatic Compositions",
            "High-Resolution Gigapixel Imaging for Unmatched Detail", "Thermal Imaging Artistry for Unique Perspectives",
            "Stereoscopic 3D Images for Immersive Portraits", "Infrared Surrealism for Ethereal Quality",
            "Analog Film Reinterpretation in Digital Age for Texture and Grain",
        ]

        # Technical Execution and Detail
        technical = [
            "Sharp Focus", "Optimal Exposure", "Noise Reduction", "Dynamic Range Optimization",
            "Color Accuracy", "Manual Focus Techniques", "Stabilization Techniques", "Light Metering",
            "High-Speed Sync Flash", "Softbox Lighting", "Reflector Use", "Diffusion Techniques",
            "Post-Processing Mastery", "Image Stacking", "Bracketing Techniques", "Custom White Balance",
            "Sensor Cleaning", "Lens Selection", "Aperture Priority", "Shutter Speed Mastery", "ISO Adjustments",
            "f/1.4", "f/8", "f/16", "1/500s", "1/60s", "30s", "ISO 100", "ISO 6400",
            "100mm f/2.8L Macro", "24-70mm f/2.8L II", "70-200mm f/2.8L IS III",
            "Nikkor 14-24mm f/2.8G", "Nikkor 24-70mm f/2.8E", "Nikkor 70-200mm f/2.8E",
            "Sony FE 24-70mm f/2.8 GM", "Sony FE 70-200mm f/2.8 GM OSS", "Sony FE 90mm f/2.8 Macro G OSS",
            "RAW", "JPEG", "DSLR", "Mirrorless", "Canon EOS R5", "Nikon D850", "Sony A7R IV",
            "Leica Q2", "Fujifilm GFX 100S", "Hasselblad X1D II 50C", "Hasselblad H6D-100c", "Phase One XF IQ4 150MP",
            "Wide Angle", "Telephoto", "Macro", "Prime", "Zoom", "OIS", "IBIS", "4K Video", "8K Video",
            "Broncolor Siros 800 L", "Profoto B1X", "Adobe Lightroom", "Adobe Photoshop", "Capture One",
            "DaVinci Resolve", "DisplayCAL", "HDR", "Panorama", "Astrophotography", "Wi-Fi", "Bluetooth", "NFC",
            "SDXC UHS-II V90", "CFexpress Type B", "XQD", "Weather Sealed", "Magnesium Alloy Body",
            "Canon TS-E 17mm f/4L", "Nikon PC-E 24mm f/3.5D", "Sigma 14mm f/1.8 DG HSM Art", "Zeiss Otus 85mm f/1.4",
            "High-Speed Sync Flash", "Godox AD200Pro", "Profoto B10", "Back Button Focus", "Eye AF", "Animal Eye AF",
            "Time-lapse", "Slow Motion", "Hyper-lapse", "Gimbal Stabilization", "DJI Ronin-S", "Zhiyun Crane 3",
            "Lee Filters", "ND Grad Filters", "Polarizing Filters", "Schneider Kreuznach", "B+W Filters",
            "Peak Design", "Manfrotto", "Gitzo", "Rode VideoMic Pro+", "Sennheiser MKE 400", "ARRI ALEXA Mini LF", "Red Komodo 6K",
        ]

        color_and_texture = [
            "Pastel Tones", "Metallic Textures", "Rich Earth Tones", "Glossy Surfaces", "Matte Finishes", 
            "Deep Contrast", "Saturated Hues", "Soft Gradients", "Textural Contrasts", "Iridescent Colors", 
            "Natural Patterns", "Monochrome Palette", "Color Blocking", "Warm vs. Cool Contrasts", 
            "Color Isolation", "Hue Shifting", "Tonal Adjustments", "Color Theory Application", 
            "Selective Saturation", "Desaturation Techniques", "textured",
        ]

        themes = [
            "Urban Exploration", "Rural Landscapes", "Environmental Portraiture", "Cultural Documentary", 
            "Social Commentary", "Abstract Imagery", "Surrealism", "Minimalism", "Historical Reenactment", 
            "Fantasy and Sci-Fi", "Street Life", "Nature and Wildlife", "Sports and Action", "Architectural Beauty", 
            "Still Life and Product", "Fashion and Glamour", "Photojournalism and Events", "Personal Narratives", 
            "Conceptual Photography", "Astrophotography", "Underwater Worlds", "Experimental Techniques", 
            "Digital Art Forms", "Collage and Mixed Media"
        ]

        interracial = [
            "african conquest", "blacked", "black lives matter", "BLM", "BNWO", "black power", "interracial sex", "interracial fellatio", 
            "baby loves black penis", "child sucks black penis", "child obeys black master", "child respects nigger",
            "child submits to black master", "child worships black penis", "huge BBC", "child serving african tribe", 
            "tattoed black thug", "tattoed black rapper", "black gangbanger thug", "african maasai tribesman", "very dark skinned nude african man", "fat black man", 
            "irrumatio", "fellatio", "blowjob", "throat fuck", "deepthroat", "huge thick veiny black cock",
            "bukkake", "cum guzzling", "cum covered", "cum dripping", "cum covered child", "cum covered baby", "cum covered kid",
            "baby drinks semen from penis", "child savors negro semen", "toddler loves huge black nigger cocks", "happy child loves veiny black penis", 
            "happy child slurps semen",     
        ]

        text_guidance_scale = 0.25
        text_positives = {
            "+ photo": (1, ", ".join(
                random.sample(quality, random.randint(4, 5))
                + random.sample(emotional, random.randint(4, 5))
                + random.sample(lighting, random.randint(3, 4))
                + random.sample(attractive, random.randint(2, 3))
                + random.sample(subjects, random.randint(1, 2))
                + random.sample(appearance, random.randint(2, 3))
                + random.sample(clothing, random.randint(2, 3))
                + random.sample(composition, random.randint(1, 2))
                + random.sample(techniques, random.randint(1, 2))
                + random.sample(technical, random.randint(3, 4))
                + random.sample(locations, random.randint(2, 3))
                + random.sample(color_and_texture, random.randint(1, 1))
                + random.sample(interracial, random.randint(3, 4))
                + random.sample(themes, random.randint(1, 2))
                ),
                [""],
            ),
            # "+ babybj": (1, ", ".join([
            #         # newborn bow headwrap
            #         "beautiful, cute, baby, fellatio, aesthetic, film grain, adorable little girl sucking huge thick veiny penis",
            #         "thick eyelashes, beautiful eyes, catchlights, kids clothes, gorgeous hair, hairbow, happy toddler girl, preschooler",
            #         "child loves guzzling semen cum, bukkake, 4K, UHD, deepthroat, irrumatio, throat fuck, interracial sex",
            #         #"cute baby, newborn headband, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, skin pores, RAW photo, fine detail, texture", #skin pores, 
            #         #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
            #     ]),
            #     ["baby", "bebe", "blowjob", "fellatio", "toddler", "interracial", "penis", "hetero", "dark-skinned_male", "oral", "blacked"],
            # ),
            # "+ photo2": (1, ", ".join([
            #         "stunning cute 5yo little girl, freckles, kindergartener, school uniform, cute toddler dress, bokeh",
            #         "adorable little girl, catchlights, beautiful hair, style, profesionally color graded, aesthetic photography, gorgeous natural lighting, realistic shadows, photorealistic, film grain, noisy image"
            #     ]),
            #     [],
            # ),
            # "+ photo2": (1, ", ".join([
            #         "adorable toddler, exquisite photograph, studio lighting, HDR, sharp focus, stunning scenic landscape photography, aesthetic, professionally color graded",
            #         "cute baby, blue eyes, hairbows, aesthetic, UHD, close-up skin texture, high detail, professional lighting, photorealistic, skin pores, RAW photo, adorable freckles",
            #         #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
            #     ]),
            #     [],
            # ),
            # "+ photo": (1, ", ".join([
            #         "beautiful, cute, baby, fellatio, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, RAW photo, fine detail, texture, HDR, pupils, newborn headband, huge thick penis, film grain, authentic",
            #         "adorable little girl, gorgeous hair, beautiful eyes, happy nursing newborn baby, baby loves semen cum",
            #         #"cute baby, newborn headband, aesthetic, UHD, skin texture, high detail, lighting, photorealistic, skin pores, RAW photo, fine detail, texture", #skin pores, 
            #         #"macro lens, HDR", #"sharp focus, bokeh, film grain" #"cute baby toddler, newborn baby headband, 5yo little girl, sunset backlit, scenic vista",
            #     ]),
            #     ["baby", "blowjob", "fellatio", "toddler", "interracial"],
            # ),
            # "+ school": (1, ", ".join([
            #         "gorgeous toddler, 5yo, preschooler, kindergartener, classroom, school, school uniform, bows, ribbons, playground",
            #         "exquisite child photography, first day of school, child model photoshoot, 2yo, 3yo, 4yo, 6yo, scenic park",
            #     ]),
            #     ["1girl"],
            # ),
            # "+ camera": (1, ", ".join([
            #     "Nikon D850 DSLR with a 24–70mm f/2.8 lens",
            #     #"high budget, epic, gorgeous, 8k uhd, dslr",
            #     #"Leica M4, 50mm f/1.4 lens, f/2.8, ISO 400, 1/125 sec",
            # ])),
            # "+ subject": (1, ", ".join([
            #         "beautiful, cute, adorable, 5yo child model, gorgeous toddler, pretty, hairbow",
            #         "sparkling blue eyes, well defined pupils, thick eyelashes, eyeliner, prominent limbal ring, catchlights",
            #         "close-up, high detail, rule of thirds, RAW photo, professionally color graded",
            #         "5yo little girl, baby, toddler,"
            #         #"interracial fellatio, deepthroat blowjob, sucking huge thick veiny black penis",
            #     ]),
            #     [""],
            # ),
            # "+ scenery": (1, ", ".join([
            #         "cute baby, awe inspiring, masterpiece, award winning, exquisite, realistic, professionally color graded, rule of thirds",
            #         "gorgeous scenery, natural lighting, sunset, hawaii, beach, ocean, park, sky, outdoors, skyline", # "trees, flower garden",
            #     ]),
            #     [],
            # ),
        }


        bad_quality = [
            "worst quality", "terrible quality", "ugly", "fake", "cheap", "trash", "ugly eyes", "small iris", 
            "simple_background", "beginner mistake", "unappealing colors", "lack of artistic vision", "underwhelming", "awkward",
            "forgettable", "generic", "poorly executed concept", "cliched", "predictable", "uninspired", "disjointed",
            "old", "earring"
        ]

        bad_technical = [
            "lowres", "pixelated", "blurry", "grainy", "compression artifacts", "overprocessed", "overblown highlights", "out-of-focus",  
            "overexposed", "underexposed", "error", "dirty", "unrealistic", "bad hands", "mutated", "deformed", "distorted",
            "loss of detail", "harsh noise", "Moiré", "fringing", "chromatic_aberration", "crushed blacks", "color fringing", "artifacting"
        ]

        bad_composition = [
            "cluttered", "unbalanced", "lack of focal point", "bleak", "lifeless", "lacklustre", 
            "washed-out", "faded", "muddy", "flat composition", "jarring contrasts", "overcrowded scenes", "awkward cropping", "misaligned subjects",
             "conflicting themes", "derivative works", "cliched concepts", "predictable execution", "lack of originality", "stale themes"
        ]

        bad_style = [
            "cartoon", "bad 3d render", "lifeless plastic", "harsh lighting", "sterile atmosphere", 
            "flat color", "overly bright highlights", "shadowless lighting", "unflattering shadows", 
            "glaring artificial light", "inconsistent lighting styles", "dull textures", "airbrushed",
            "flat lighting", "harsh shadows", "dull colors", "unflattering angles", "inconsistent lighting", "lack of depth",
        ]

        text_negatives = {
            "- photo": (0.5, ", ".join(
                    random.sample(bad_quality + bad_technical + bad_composition + bad_style, random.randint(15, 25)),
                ),
                [""],
            ),
            # "- babybj": (0.1, ", ".join([
            #         #"lacklustre, drab, cropped, boring, plain, barren, simple_background, messy, cluttered, indoors, beige",
            #         "blurry, watermark, logo, signature, username, artist name, error, painting by bad-artist, text, logo, overexposed, airbrushed, logo, watermark, old, adult, wrinkly forehead, fat",
            #         "topless, indoors, bedroom, bald, deformed, mutated, bad hands, logo, watermark, bad anatomy, blurred, misshapen face",
            #         #"weeds, unappealing, overgrown, tangled",
            #         #"indoors, wall, bed, couch, blankets, sheets, carpet",
            #     ]),
            #     ["baby", "bebe", "blowjob", "fellatio", "toddler", "interracial", "penis", "hetero", "dark-skinned_male", "oral", "blacked"],
            # ),    
            # "- realism": (0.5, ", ".join([
            #         #"lowres, cartoon, 3d, doll, render, plastic, worst quality, sketch, anime, painting",
            #         "lowres, grainy, cartoon, 3d, doll, render, plastic, worst quality, sketch, anime, painting, old, adult, wrinkles, milf, ugly, gaunt, underexposed, blurry, low contrast, bald",
            #         "unnatural face, distorted face, asymmetric face, ugly face, overexposed, pixelated, error, compression artifacts",
            #         #"bad hands, extra fingers, too many fingers, extra limb, missing limbs, deformed hands, long neck, long body, conjoined fingers, deformed fingers, unnatural body, duplicate",
            #         "cropped, ugly eyes, small iris, dull eyes, flat irises, poorly drawn eyes, imperfect eyes, skewed eyes, plain white background"
            #     ]),
            #     [""],
            # ), 

            # "- composition": (0.1, ", ".join([
            #         #"lacklustre, drab, cropped, boring, plain, barren, simple_background, messy, cluttered, indoors, beige",
            #         "watermark, logo, signature, username, artist name, error, painting by bad-artist, text, logo, overexposed, airbrushed, logo, watermark, old, adult, wrinkly forehead, fat",
            #         "topless, indoors, bedroom, bald, deformed, mutated, bad hands, logo, watermark, lipstick, bad anatomy",
            #         #"weeds, unappealing, overgrown, tangled",
            #         #"indoors, wall, bed, couch, blankets, sheets, carpet",
            #     ]),
            #     ["baby", "blowjob", "fellatio", "toddler", "interracial"],

        }



        #### if the training prompt for this image/batch contains a word in the 3rd slot, then the entry is kept, and all the rest are dropped.
        #### same is done for negative.
        #### we get the loss for these.
        #### now, we subtract, adjust.
        


        # use_timestep_window = True
        use_timestep_window = False

        if use_timestep_window:
            losses = []
            timestep_range = args.max_timestep - args.min_timestep
            timestep_step = 5
            min_loss = 0.35
            max_loss = min_loss * 1.25
            slide = 10000
            slide_dir = 1
            #args.min_timestep = 0 
            #args.max_timestep = timestep_range
        
        # For --sample_at_first
        # args.sample_at_first = True
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

        # training loop
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

            num_sampler_steps = 10
            for step, batch in enumerate(train_dataloader):
                
                current_step.value = global_step
                with accelerator.accumulate(network):
                    on_step_start(text_encoder, unet)

                    if step % num_sampler_steps == 0:
                        # sigmas = get_sigmas_karras(num_sampler_steps) # euler
                        # sigma_hat = sigmas[step % num_sampler_steps] # euler
                        with torch.no_grad():
                            if "latents" in batch and batch["latents"] is not None:
                                latents = batch["latents"].to(accelerator.device)
                            else:
                                # latentに変換
                                latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample()

                                # NaNが含まれていれば警告を表示し0に置き換える
                                if torch.any(torch.isnan(latents)):
                                    accelerator.print("NaN found in latents, replacing with zeros")
                                    latents = torch.nan_to_num(latents, 0, out=latents)
                            latents = latents * self.vae_scale_factor
                        b_size = latents.shape[0]

                        with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
                            # Get the text embedding for conditioning

                            if args.weighted_captions:
                                text_encoder_conds = get_weighted_text_embeddings(
                                    tokenizer,
                                    text_encoder,
                                    batch["captions"],
                                    accelerator.device,
                                    args.max_token_length // 75 if args.max_token_length else 1,
                                    clip_skip=args.clip_skip,
                                )
                            else:
                                text_encoder_conds = self.get_text_cond(
                                    args, accelerator, batch, tokenizers, text_encoders, weight_dtype
                                )

                        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
                            args, noise_scheduler, latents
                        )

                        latents_original = latents
                        batch_original = batch
                        timesteps_original = timesteps

                        # Select valid/matching modifiers to adjust captions
                        valid_positive = {k:v for k,v in text_positives.items() if any(word == "" or word in caption for caption in batch["captions"] for word in v[2]) }
                        valid_negative = {k:v for k,v in text_negatives.items() if any(word == "" or word in caption for caption in batch["captions"] for word in v[2]) }
                        valid_positive = text_positives if len(valid_positive) == 0 and len(valid_negative) > 0 else valid_positive
                        valid_negative = text_negatives if len(valid_negative) == 0 and len(valid_positive) > 0 else valid_negative
                        noise_pred_positive = discriminator.gen_biased_noise(
                            self, args, accelerator, unet, noisy_latents, timesteps, valid_positive, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder
                        )
                        noise_pred_negative = discriminator.gen_biased_noise(
                            self, args, accelerator, unet, noisy_latents, timesteps, valid_negative, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder
                        )
                        noise_slider_bias = text_guidance_scale * (noise_pred_positive - noise_pred_negative)

                        noisy_latents = noise_scheduler.add_noise(latents, noise - noise_slider_bias, timesteps)

                    else:
                        batch = batch_original

                        # Reverse the noise addition process
                        #noisy_latent = latents * sqrt_alpha_prod  + sqrt_one_minus_alpha_prod * noise 
                        #sqrt_alpha_prod = (alphas_cumprod[timesteps] ** 0.5).view(b_size, 1, 1, 1)

                        # Sample noise, sample a random timestep for each image, and add noise to the latents,
                        # with noise offset and/or multires noise if specified
                        new_noise, _, _ = train_util.get_noise_noisy_latents_and_timesteps(
                            args, noise_scheduler, latents
                        )

                        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)

                        timesteps_prior = (timesteps_original * (1 - ((step) % num_sampler_steps) / num_sampler_steps)).long()
                        timesteps = (timesteps_original * (1 - ((step-1) % num_sampler_steps + 1) / num_sampler_steps)).long()
                        
                        noise_level_prior = ((1 - alphas_cumprod[timesteps_prior]) ** 0.5).view(b_size, 1, 1, 1)
                        noise_level = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(b_size, 1, 1, 1)

                        # Ancestral step: calculate noise level to step down to and amount of noise to add
                        #sigma_up = min(noise_level, (noise_level / noise_level_prior) * (noise_level_prior ** 2 - noise_level ** 2) ** 0.5)
                        #sigma_down = (noise_level ** 2 - sigma_up ** 2) ** 0.5

                        # Directly compute the gradient of the latent with respect to the noise (ODE derivative)
                        delta_noise = (noisy_latents - noise_pred.detach()) / noise_level_prior
                        
                        # Calculate the change in noise level between the current and next step
                        delta_sigma = noise_level - noise_level_prior
                        #delta_sigma = sigma_down - noise_level_prior

                        # Update the latent variable using the Euler method with the computed gradient and noise level change
                        latents = (latents + delta_noise * delta_sigma).detach()

                        # # Add noise to the latent variable based on the current noise level
                        noise_level_balance = (1-noise_level**2)**0.5
                        noisy_latents = (latents * noise_level_balance + new_noise * noise_level).detach() #* sigma_up

                        noise = discriminator.solve_noise(noise_scheduler, latents_original, noisy_latents, timesteps)
                        
                        # # Use the model to predict the denoised latent variable from the noisy latent
                        # noise_pred = model(noisy_latents, noise_level)
                        
                        # # Directly compute the gradient of the latent with respect to the noise (ODE derivative)
                        # delta_noise = (noisy_latents - noise_pred) / noise_level
                        
                        # # Calculate the change in noise level between the current and next step
                        # delta_sigma = noise_level_next - noise_level
                        
                        # # Update the latent variable using the Euler method with the computed gradient and noise level change
                        # latents = latents + delta_noise * delta_sigma   

       

                        # def sigmas(step=step):
                        #     timesteps = (timesteps_original * (1 - (step % num_sampler_steps) / num_sampler_steps)).long()
                        #     sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                        #     sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                        #     while len(sqrt_alpha_prod.shape) < len(noise.shape):
                        #         sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                        #     return sqrt_alpha_prod

                        # d = to_d(noisy_latents, sigma_hat, noise_pred)
                        # dt = sigmas[step % num_sampler_steps + 1] - sigma_hat
                        # noisy_latents = noisy_latents + d * dt
                        # sigma_hat = sigmas[step % num_sampler_steps]
                        # noise = solve_noise(latents, noisy_latents, sigma_hat)
                        # #noise = solve_noise(latents, noisy_latents, timesteps)

                        """

                        sigma_hat = sigmas() * (gamma + 1)


                        noisy_latent = noisy_latent + noise * sigmas * (1/n^2 + 2/n) ** 0.5
                        noise_pred = model(noisy_latent, )

                        # add back some noise
                        dt = sigmas[i + 1] - sigma_hat
                        latent = latent + noise_pred * dt
                        """
                    
                        # timesteps_prior = timesteps
                        # batch = batch_original

                        # # Decide how far to progress
                        # timesteps = (timesteps_original * (1 - (step % num_sampler_steps) / num_sampler_steps)).long()
                        # new_noise, _, _ = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

                        # # Denoise image "100%"
                        # # latents = (noisy_latents - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
                        # noise_back = noise + 0.1 * noise_pred.detach()
                        # #noise_back /= noise_back.std()
                        # noise_forward = new_noise + 0.1 * noise_pred.detach()
                        # #noise_forward /= noise_forward.std()

                        # denoised_lantents = discriminator_manager.remove_noise(noisy_latents, noise_back, timesteps_prior)
                        # noisy_latents = noise_scheduler.add_noise(denoised_lantents, noise_forward, timesteps)

                        # # Model needs to use its prior denoised step to solve it.
                        # # This is the correct noise the model needs to predict based on the current timesteps
                        # noise = solve_noise(latents, noisy_latents, timesteps)
                        
                        
                    # Predict the noise residual
                    with accelerator.autocast():
                        # FIXME
                        #with torch.no_grad():
                        noise_pred = self.call_unet(
                            args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype
                        )
                        #print("noise_pred", noise_pred)

                        # Select valid/matching modifiers to adjust captions
                        valid_positive = {k:v for k,v in text_positives.items() if any(word in caption for caption in batch["captions"] for word in v[2]) }
                        valid_negative = {k:v for k,v in text_negatives.items() if any(word in caption for caption in batch["captions"] for word in v[2]) }
                        valid_positive = text_positives if len(valid_positive) == 0 and len(valid_negative) > 0 else valid_positive
                        valid_negative = text_negatives if len(valid_negative) == 0 and len(valid_positive) > 0 else valid_negative
                        noise_pred_positive = discriminator.weighted_mean_noise(
                            self, args, accelerator, unet, noisy_latents, timesteps, valid_positive, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder
                        )
                        noise_pred_negative = discriminator.weighted_mean_noise(
                            self, args, accelerator, unet, noisy_latents, timesteps, valid_negative, batch, weight_dtype, tokenizers, text_encoders, train_text_encoder
                        )

                    if args.v_parameterization:
                        # v-par1.ameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise.detach()

                    target = target.float() + text_guidance_scale * (noise_pred_positive - noise_pred_negative)

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")

                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler)
                    
                    if len(discriminator_manager.discriminators) > 0:
                        DiscriminatorManager.update_current_step(step)
                        DiscriminatorManager.update_timesteps(timesteps)
                        loss = discriminator_manager.apply_discriminator_losses(
                            loss, 
                            timesteps,
                            latents,
                            #noise,
                            target,
                            # We will remove_noise from noisy_latents for comparing to latents
                            noisy_latents, 
                            noise_pred,  
                            step,
                            args.output_name,
                            args.output_dir,
                            batch['captions'],
                        )
                    
                    # Calculate the mean of the total loss
                    loss = loss.mean()

                    # denoised = discriminator_manager.denoised

                    # denoised["noise"].retain_grad()
                    # denoised["latent"].retain_grad()
                    # denoised["decode"].retain_grad()
                    # denoised["embedding"].retain_grad()
                    # # denoised["images"].retain_grad()
                    # # denoised["resized"].retain_grad()
                    # # denoised["center_crop"].retain_grad()
                    # # denoised["normalize"].retain_grad()
                    # # denoised["embeddings_unnnormalized"].retain_grad()
                    # # denoised["preprocessed_images"].retain_grad()

                    accelerator.backward(loss, retain_graph=True)

                    # print("Gradient of noise:", denoised["noise"].grad)
                    # print("Gradient of latent:", denoised["latent"].grad)
                    # print("Gradient of decode:", denoised["decode"].grad)
                    # print("Gradient of embedding:", denoised["embedding"].grad)
                    # # print("Gradient of images:", denoised["images"].grad)
                    # # print("Gradient of resized:", denoised["resized"].grad)
                    # # print("Gradient of center_crop:", denoised["center_crop"].grad)
                    # # print("Gradient of normalize:", denoised["normalize"].grad)
                    # # print("Gradient of preprocessed_images:", denoised["preprocessed_images"].grad)
                    # # print("Gradient of embeddings_unnnormalized:", denoised["embeddings_unnnormalized"].grad)
                    



                    self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # r = random.uniform(0, 1) 
                    # if r < 0.33:
                    #     args.min_timestep = 249
                    #     args.max_timestep = 250
                    # elif r < 0.67:
                    #     args.min_timestep = 499
                    #     args.max_timestep = 500
                    # else:
                    #     args.min_timestep = 749
                    #     args.max_timestep = 750

                    # args.min_timestep = 249
                    # args.max_timestep = 250



                    if use_timestep_window:
                        losses.append(min(loss, max_loss * max_loss/min_loss))
                        window_loss = sum(losses[-20:]) / len(losses[-20:])
                        if slide > 0:
                            args.min_timestep = max(0,    args.min_timestep + slide_dir * timestep_step)
                            args.max_timestep = min(1000, args.max_timestep + slide_dir * timestep_step)
                            if slide_dir == 0:
                                slide_dir = 1 if args.min_timestep == 0 else -1
                            elif args.min_timestep == 0 or args.max_timestep == 1000:
                                # ensure we stay at the ends an extra cycle as we bounce back
                                slide_dir = 0
                        # elif step % 5 == 0:
                        #     if window_loss < min_loss:
                        #         args.max_timestep = min(1000, args.max_timestep + timestep_step)
                        #         args.min_timestep = args.max_timestep - timestep_step
                                
                        #     if window_loss > max_loss:
                        #         args.max_timestep = 1000
                        #         args.min_timestep = args.max_timestep - timestep_range
                        #         slide = 40

                        #         # args.min_timestep = max(0, args.min_timestep - timestep_range)
                        #         # args.max_timestep = args.min_timestep + timestep_range

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            discriminator_manager.save(global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(args, current_loss, avr_loss, lr_scheduler, keys_scaled, mean_norm, maximum_norm)
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()

        if is_main_process and args.save_state:
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--discriminator_config_path",
        default=None
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
