import json
from typing import Any
from loguru import logger
from torch.utils.data import Dataset
import os
import random
from transformers import AutoTokenizer, BitsAndBytesConfig


class SFTDataset_jnl(Dataset):
    # 读取一个数据集下的数据
    # 
    def __init__(self, path, tokenizer, max_seq_length, noise_tp = "cut", is_train = True, type = "NER"):
        self.type = type
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.path = path
        self.noise_tp = noise_tp
        self.max_len = max_seq_length
        self.example = self.load_dataset(self.path)
        logger.info("there are {} data in dataset".format(len(self.example)))
        self.padding = True
        
    def load_dataset(self, path):
        example = []
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                try:
                    line = json.loads(line)
                    instruction = "Please list all entity words in the text that fit the category.Output format is \"type1: word1; type2: word2\". \nOption: cell line, protein, RNA, DNA, cell type, None"
                    # "Please find all the entity words associated with the category in the given text.Output format is \"type1: word1; type2: word2\". \nOption: cell line, protein, RNA, DNA, cell type, None \n"
                    sentence = line["input"]
                    output = " " + line["output"]
                    example.append({
                        "id": str(idx),
                        "sentence": sentence,
                        "label": output,
                        "instruction": instruction,
                    })
                except:
                    continue
        return example
    
    def __len__(self):
        return len(self.example)
    
    def __getitem__(self, index):
        data = self.example[index]
        instruction = data["instruction"] +" \n" + "Text: " + "{0}" + " \n" + "Answer:"
        instruction = instruction.format(data["sentence"])
        noise_instruction = data["instruction"] +" \n" + "Text: " + "{0}" + " \n" + "Answer:"
        noise_instruction = noise_instruction.format(data["sentence"])
        label = data["label"]
        if index == 0:
            print(instruction + label)

        model_inputs = self.tokenizer(
                instruction,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )
        
        noise_inpts = self.tokenizer(
                noise_instruction,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )

        labels = self.tokenizer(
                label,
                max_length=self.max_len,
                padding=self.padding,
                truncation=True,
            )

        tokenized_input = model_inputs["input_ids"] + [self.eos_token_id]
        tokenized_noise = noise_inpts["input_ids"] + [self.eos_token_id]
        tokenized_label = labels["input_ids"][1:] + [self.eos_token_id]

        input_ids = tokenized_input + tokenized_label
        input_ids = input_ids[:self.max_len]
        test_input_ids = tokenized_input[:self.max_len]
        
        target_mask = [0] * min(self.max_len, len(tokenized_input)) + [1] * max(min(self.max_len - len(tokenized_input), len(tokenized_label)), 0)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        
        noise_ids = tokenized_noise + tokenized_label
        noise_ids = noise_ids[:self.max_len]
        noise_mask = [0] * min(self.max_len, len(tokenized_noise)) + [1] * max(min(self.max_len - len(tokenized_noise), len(tokenized_label)), 0)
        assert len(noise_ids) == len(noise_mask)
        

        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        noise_att_mask = [1] * len(noise_ids)
        assert len(noise_ids) == len(noise_mask) == len(noise_att_mask)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
            'noise_ids': noise_ids,
            'noise_att_mask': noise_att_mask,
            'noise_mask': noise_mask,
            'test_input': test_input_ids,
            'label_ids': tokenized_label
        }
        return inputs            
                
    