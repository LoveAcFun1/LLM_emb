from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    AutoModelForCausalLM
)
import transformers
from utils.Drop_Llama import Drop_Llama
import argparse
from loguru import logger
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass, field

from component.collator import SFTDataCollator
from component.dataset import  ChatGLM2SFTDataset
from utils.dataset_new import SFTDataset, SFTDataset_all
# from utils.dataset_BIO import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer
from component.loss import TargetLMLoss
from component.llama_model import Llama_seq2seq
from utils.metrics import get_metrics
import json
import logging
import os
from typing import List, Optional, Tuple, Union
from utils.Drop_ATT_Llama import LlamaForCausalLM
from utils.Drop_ATT_gemma import GemmaForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.llava_trainer import LLaVATrainer


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model, quantization: Optional[int] = None):
    if quantization is None:
        cls = torch.nn.Linear
    elif quantization == 4:
        from bitsandbytes.nn import Linear4bit

        cls = Linear4bit
    elif quantization == 8:
        from bitsandbytes.nn import Linear8bitLt

        cls = Linear8bitLt
    else:
        raise ValueError(f"Unknown quantization type: {quantization}")

    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/qlora/qwen-7b-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # logger.add(join(training_args.output_dir, 'train.log'))
    # logger.info("train_args:{}".format(training_args))
    training_args.model_max_length = args.max_seq_length
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    # 下面的设置至关重要，否则无法多卡训练
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # device_map = "auto"
    # # if we are in a distributed setting, we need to set the device map and max memory per device
    # if os.environ.get('LOCAL_RANK') is not None:
    #     local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    #     device_map = {'': local_rank}

    training_args.ddp_find_unused_parameters = False
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    
    quantization = args.quantization
    torch_dtype = args.torch_dtype
    quant_args = {}
    torch_dtype = torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)
    
    if quantization is not None:
        quant_args = {"load_in_4bit": True} if quantization == 4 else {"load_in_8bit": True}
        if quantization == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch_dtype in ["auto", None] else torch_dtype,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
    else:
        # logging.info(f"Loading model with dtype: {torch_dtype}")
        bnb_config = None
        
    config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            pretraining_tp=1,  # Fix mat1 and mat2 shapes cannot be multiplied  error with LLaMA-2
            # See https://github.com/huggingface/transformers/pull/24906
        )    

    # 加载模型
    # model = Drop_Llama.from_pretrained(
    #         pretrained_model_name_or_path=args.model_name_or_path,
    #         device_map=device_map,
    #         quantization_config=bnb_config,
    #         torch_dtype=torch_dtype,
    #         config=config,
    #         trust_remote_code=True,
    #         **quant_args,
    #     )
    if args.use_ptuning:
        config.rms_norm_eps = 1e-5 
        model = LlavaLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            # cache_dir=None,
            config=config,
            trust_remote_code=True,
            **quant_args,
        )
        model.config.use_cache = False
    else:
        model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_name_or_path,
                device_map=device_map,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                config=config,
                trust_remote_code=True,
                **quant_args,
            )
    # model = GemmaForCausalLM.from_pretrained(
    #         pretrained_model_name_or_path=args.model_name_or_path,
    #         device_map=device_map,
    #         quantization_config=bnb_config,
    #         torch_dtype=torch_dtype,
    #         config=config,
    #         trust_remote_code=True,
    #         **quant_args,
    #     )
    
    
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
    # 指加载训练集
    if model.config.model_type == 'chatglm':
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    else:
        train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, type = args.task)
        eval_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, is_train = False, type = args.task, rate = 1.0)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)

    # casts all the non int8 modules to full precision (fp32) for stability
    if args.quantization is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    else:
        model.gradient_checkpointing_enable()
        
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model, quantization=args.quantization)
    # 初始化lora配置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    if args.use_ptuning:
        model.get_model().initialize_prompt_modules(
            model_args=args,
            fsdp=None
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # data_args.image_processor = vision_tower.image_processor
        args.is_multimodal = True

        model.config.image_aspect_ratio = args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
        if args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = True
        if model.config.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        # if args.quantization in [4, 8]:
        #     model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = args.mm_use_im_start_end = args.mm_use_im_start_end
        model.config.mm_projector_lr = None
        training_args.use_im_start_end = args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
        # model.initialize_vision_tokenizer(args, tokenizer=tokenizer)
    
    if args.quantization in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    loss_func = TargetLMLoss(ignore_index=-100, tokenizer = tokenizer)

    compute_metrics = get_metrics(tokenizer, args.task)

    # 初始化Trainer
    if args.use_ptuning:
        trainer = LoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset = eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_loss=loss_func,
            compute_metrics = compute_metrics,
        )
    else:
        trainer = LoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset = eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_loss=loss_func,
            compute_metrics = compute_metrics,
        )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    train_result = trainer.train()
    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


