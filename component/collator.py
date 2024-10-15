from typing import Any, Dict, List
import torch


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length, is_test = False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.is_test = is_test

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch]
        test_len = [len(x['test_input']) for x in batch]
        label_len = [len(x['labels']) for x in batch]
        prompt_len = [len(x['p_prompt']) for x in batch]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        batch_max_testlen = min(max(test_len), self.max_seq_length)
        batch_max_labellen = min(max(label_len), self.max_seq_length)
        batch_max_pmtlen = min(max(prompt_len), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, pmt_batch, target_mask_batch, test_batch, labels_batch = [], [], [], [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            labels = x["labels"]
            p_prompt = x["p_prompt"]
            test_input = x["test_input"]
            label_ids = x["label_ids"]
            target_mask = x['target_mask']

            padding_len = batch_max_len - len(input_ids)
            test_padlen = batch_max_testlen - len(test_input)
            label_padlen = batch_max_labellen - len(labels)
            pmt_padlen = batch_max_pmtlen - len(p_prompt)
            
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            p_prompt = p_prompt + [0] * pmt_padlen
            test_input =   [0] * test_padlen + test_input   ###oioioio
            target_mask = target_mask + [0] * label_padlen
            # label_ids = label_ids + [0] * label_padlen
            labels = labels + [-100] * label_padlen

            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            p_prompt = p_prompt[:self.max_seq_length]
            test_input = test_input[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]
            # label_ids = label_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]


            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pmt_batch.append(p_prompt)
            target_mask_batch.append(target_mask)
            # test_labels_batch.append(label_ids)
            test_batch.append(test_input)
            labels_batch.append(labels)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        pmt_batch = torch.tensor(pmt_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        # test_labels_batch = torch.tensor(test_labels_batch, dtype=torch.long)
        test_batch = torch.tensor(test_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'test_labels': labels_batch,
            'target_mask': target_mask_batch,
            # 'test_labels':test_labels_batch,
            'test_ids':test_batch,
            'noise_pos' : noise_batch,
            'normal_pos': normal_batch,
            # 'ins_mask':ins_mask_batch
        }
        
        inputs["images"] = pmt_batch
        return inputs
