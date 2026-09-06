import torch.nn as nn
from transformers import BertModel

class BertRNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.rnn = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_num_layers,
            batch_first=True,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier = nn.Linear(config.rnn_hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.rnn_dropout)

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = bert_output.last_hidden_state  # (batch, seq_len, hidden)

        # RNN
        rnn_out, _ = self.rnn(seq_out)  # (batch, seq_len, rnn_hidden)

        # 取最后一个时间步
        last_out = rnn_out[:, -1, :]
        last_out = self.dropout(last_out)
        logits = self.classifier(last_out)
        return logits