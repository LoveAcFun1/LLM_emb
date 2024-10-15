from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})


@dataclass
class QLoRAArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    train_file: str = field(metadata={"help": "训练集"})
    model_name_or_path: str = field(metadata={"help": "预训练权重路径"})
    task_type: str = field(default="", metadata={"help": "预训练任务：[sft, pretrain]"})
    eval_file: Optional[str] = field(default="", metadata={"help": "the file of training data"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    add_nosie: Optional[bool] = field(default=False, metadata={"help": "add noise"})
    task: Optional[str] = field(default="", metadata={"help": "the task name RE/NER"})
    quantization: Optional[int] = field(default=None, metadata={"help": "4/8/None"})
    torch_dtype: Optional[str] = field(default="float16", metadata={"help": "the type in model, float16/bfloat16"})
    use_ptuning: Optional[bool] = field(default=True, metadata={"help": "True/False"})
    vision_tower: Optional[str] = field(default="clip-vit-large-patch14", metadata={"help": "the type in model, float16/bfloat16"})
    mm_vision_select_layer: Optional[int] = field(default=-2, metadata={"help": "4/8/None"})
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

