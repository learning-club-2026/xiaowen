import torch

class Config:
    # BERT
    bert_model_name = 'prajjwal1/bert-tiny'    
    max_seq_len = 128

    # RNN
    rnn_hidden_size = 128
    rnn_num_layers = 2
    rnn_dropout = 0.3

    # Training
    batch_size = 32
    epochs = 5
    learning_rate = 2e-5

    # Label mapping: 1~2星->负(0), 3星->中(1), 4~5星->正(2)
    num_classes = 3

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Random seed
    seed = 42
