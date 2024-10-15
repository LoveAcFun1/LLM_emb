import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class PromptTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.hidden_size = 4096
        self.LSTM = torch.nn.LSTM(4096, 4096, 2, bidirectional=True, batch_first=True)
        self.linner = nn.Linear(self.hidden_size, self.hidden_size)

    @torch.no_grad()
    def forward(self, images):
        

        images = images.to(dtype=torch.float16)
        out, (hn, cn)  = self.LSTM(images)
        ins_emb = hn.permute(1, 0, 2)
        ins_emb = self.linner(ins_emb).to(dtype=torch.float32)
        # 无参数更新
        # ins_emb = torch.max(images, 1)
        # ins_emb = ins_emb[0].unsqueeze(1).to(dtype=torch.float32)
        # 线性层更新
        # ins_emb = ins_emb[0].unsqueeze(1).to(dtype=torch.float16)
        # ins_emb = self.linner(ins_emb)
        return ins_emb

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    # @property
    # def dtype(self):
    #     return self.vision_tower.dtype

    # @property
    # def device(self):
    #     return self.vision_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    # @property
    # def num_patches_per_side(self):
    #     return self.config.image_size // self.config.patch_size

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2

