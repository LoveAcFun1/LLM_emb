import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.AT_Drop import ADdrop_Loss

class Loss(object):
    """
    所有loss的类父类
    """
    def __call__(self, model, inputs, training_args, return_outputs=False):
        """
        todo label smoothing
        用于计算loss。
        看源码发现，return_outputs=True为train时调用，return_outputs=False为eval和predict调用
        :param model: 模型
        :param inputs: 模型输入，dict
        :param training_args: 训练配置参数
        :param return_outputs:是否返回模型的输出
        :return:
        """
        raise NotImplemented


class TargetLMLoss(Loss):

    def __init__(self, ignore_index, tokenizer):
        super().__init__()
        # self.loss_fn = ADdrop_Loss(ignore_index)
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs, training_args, return_outputs=False, generation_config = None):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        if generation_config is None:
            # 模型前馈预测
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images = inputs["images"], return_dict=True)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

            # # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
            # labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # loss, outputs, labels = self.loss_fn(model, inputs, return_outputs)
            
        else: 
            test_input = inputs["test_ids"]
            generated_tokens = model.generate(
                                    inputs=test_input, max_new_tokens=generation_config.max_new_tokens, do_sample=True,
                                    top_p=generation_config.top_p, temperature=generation_config.temperature, repetition_penalty=generation_config.repetition_penalty,
                                    eos_token_id=generation_config.eos_token_id, images = inputs["images"]#,attention_mask = inputs["test_mask_batch"]
                                )
            loss = None
            
            # for out in generated_tokens:
            #     out = out.tolist()
            #     response = self.tokenizer.decode(out)
            #     response = response.strip().replace(self.tokenizer.eos_token, "").strip()
            #     print("Firefly：{}".format(response))
                
            return (loss, generated_tokens, inputs["test_labels"]) if return_outputs else loss
        return (loss, outputs, labels) if return_outputs else loss
