from transformers import AutoTokenizer
import torch

import sys
sys.path.append("../../")
from component.utils import ModelUtils
import os
import json
"""
单轮对话，不具有对话历史的记忆功能
"""

def _load_dataset(labels_path):
    with open(labels_path, encoding="utf-8") as labels_f:
        labels = json.load(labels_f)

    return labels
def main():
    # 使用合并后的模型进行推理
    # model_name_or_path = '/home/sda/xuguangtao/Firefly-master/checkpoint/firefly-qwen-7b-qlora-sft-merge'
    # adapter_name_or_path = None

    # 使用base model和adapter进行推理，无需手动合并权重
    model_name_or_path = '/home/lqb/llama-7B'
    adapter_name_or_path = None

    # 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
    load_in_4bit = False
    # 生成超参配置
    max_new_tokens = 50
    
    top_p = 0.9
    temperature = 0.05
    repetition_penalty = 1.0
    device = 'cuda'
    # 加载模型
    model = ModelUtils.load_model(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id

    text = input('User：')
    while True:
        text = text.strip()
        # chatglm使用官方的数据组织格式
        if model.config.model_type == 'chatglm':
            text = '[Round 1]\n\n问：{}\n\n答：'.format(text)
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
        else:            
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
            eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
            pad_token_id = torch.tensor([[0,0,0,0,0,0]], dtype=torch.long).to(device)
            input_ids = torch.concat([pad_token_id, bos_token_id, input_ids, eos_token_id], dim=1)
            
           
            embeds = model.get_input_embeddings()(input_ids)
            attention_mask = torch.where(input_ids == 0, torch.tensor(0.0), torch.tensor(1.0)).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                # inputs_embeds=embeds, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
            outputs_emb = model.generate(
                # input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                inputs_embeds=embeds, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id, attention_mask = attention_mask
            )
        for out in outputs:
            out = out.tolist()[len(input_ids[0]):]
            response = tokenizer.decode(out)
            response = response.strip().replace(tokenizer.eos_token, "").strip()
            print("Firefly：{}".format(response))
        for out in outputs_emb:
            out = out.tolist()
            response = tokenizer.decode(out)
            response = response.strip().replace(tokenizer.eos_token, "").strip()
            print("Firefly_emb：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()
