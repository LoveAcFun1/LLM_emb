from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
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

tokenizer = AutoTokenizer.from_pretrained(
        "/home/lqb/LLMs/LLama3-8B",
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )

print(tokenizer.pad_token_id)

s = '<|end_of_text|>'