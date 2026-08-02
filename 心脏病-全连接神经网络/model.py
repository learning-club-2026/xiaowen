import torch.nn as nn

class FCNN(nn.Module):
    """
    全连接神经网络 (Full Connected Neural Network)
    结构: 
        - 输入层: INPUT_DIM 个神经元
        - 隐藏层1: HIDDEN_DIM1 个神经元 + ReLU 激活 + Dropout
        - 隐藏层2: HIDDEN_DIM2 个神经元 + ReLU 激活 + Dropout
        - 输出层: 1 个神经元 + Sigmoid 激活（输出概率）
    """
    def __init__(self, input_dim, hidden1, hidden2, output_dim, dropout_rate):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x