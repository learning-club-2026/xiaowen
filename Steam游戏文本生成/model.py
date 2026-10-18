"""基于预训练 T5 的结构化数据条件文本生成模型。"""
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM


class SteamTextGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        return self.seq2seq(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def save_pretrained(self, save_directory):
        self.seq2seq.save_pretrained(save_directory)
