import math

import torch
import torch.nn.functional as F
from torch import nn as nn

class ADdrop_Loss(object):
    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def loss(self, outputs, input_ids, target_mask):
        loss = None
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., 1:-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, labels
    
    def kl_loss(self, logits1, logits2, loss):
        p = torch.log_softmax(logits1.view(-1, self.num_labels), dim=-1)
        p_tec = torch.softmax(logits1.view(-1, self.num_labels), dim=-1)
        q = torch.log_softmax(logits2.view(-1, self.num_labels), dim=-1)
        q_tec = torch.softmax(logits2.view(-1, self.num_labels), dim=-1)

        kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum()
        reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum()
        loss += 0.5 * (kl_loss + reverse_kl_loss) / 2.
        return loss
    
    def LSVR_loss(self, a):
        # 计算奇异值分解
        _, s, _ = torch.svd(a)
        # 计算奇异值的和
        sum_sigma = torch.sum(s, -1)
        # 获取每个奇异值的指数
        exp_sigma, _ = torch.max(s, -1)
        # 计算损失值
        loss = -torch.log(exp_sigma / sum_sigma)
        # 计算损失值的平均值
        avg_loss = torch.mean(loss)
        return avg_loss
        
        
    def __call__(self, model, inputs, return_outputs = True):
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        # get outputs1
        # noise = torch.zeros([input_ids.size(0),input_ids.size(1),4096]).to(input_ids.device)
        # noise = torch.zeros([input_ids.size(0),32,input_ids.size(1),input_ids.size(1)]).to(input_ids.device)
        # noise.requires_grad_()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        loss, labels = self.loss(outputs, input_ids, target_mask)
        # 奇异值损失
        # logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        # loss += 0.1 * self.LSVR_loss(logits)
        
        # 对抗训练部分
        # loss.backward(retain_graph=True)
        # delta_grad = noise.grad #对噪声向量计算梯度
        # norm = delta_grad.norm()
        # # model.zero_grad()
        
        # if (torch.isnan(norm) or torch.isinf(norm)):  #会出现nan值的处理，这里我觉得很有必要，因为我自己搞的时候就发现很容易出现nan值
        #     # skim this batch
        #     return loss, outputs, labels
        # # get outputs2
        
        # new_noise = noise + (delta_grad / norm) * 2e-5  #更新噪声向量
        # adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, noise = new_noise)
        # adv_loss, labels = self.loss(adv_outputs, input_ids, target_mask)
        # loss += adv_loss
        
        return loss, outputs, labels