import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel

class Embedding_ins(torch.nn.Module):
    def __init__(self, model, vocab_size, hidden_size):
        super().__init__()
        self.model = model
        self.LSTM = torch.nn.LSTM(4096, 4096, 1, batch_first=True)
        self.linear = torch.nn.Linear(4096, 4096)
        
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,):
        
        inputs_emb = self.model.get_input_embeddings()(input_ids)
        out, (hn, cn)  = self.LSTM(inputs_emb)
        ins_emb = self.linear(hn).permute(1, 0, 2)
        
        embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ins_emb, embeds], dim=1)
        
        attention_mask = torch.cat([attention_mask[:,0].unsqueeze(1), attention_mask], dim=1)
        
        return self.model(input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                        noise = noise)
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def generate(self,
            input_ids,
            max_new_tokens,
            do_sample,
            top_p,
            temperature,
            repetition_penalty,
            eos_token_id,
            **kwargs):
        
        inputs_emb = self.model.get_input_embeddings()(input_ids).float()
        out, (hn, cn)  = self.LSTM(inputs_emb)
        ins_emb = self.linear(hn).permute(1, 0, 2)
        
        embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([ins_emb, embeds], dim=1).half()
        
        return self.model.generate(
                                    inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, do_sample=True,
                                    top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                                    eos_token_id=eos_token_id, **kwargs
                                )
        
    def save_pretrained(
                self, output_dir, state_dict, safe_serialization
            ):
        return self.model.save_pretrained(output_dir, state_dict = state_dict, safe_serialization = safe_serialization)