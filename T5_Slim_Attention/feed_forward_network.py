import torch.nn as nn

"""
FFN: dense_layer(dim=d_ff) -> activation function -> dense_layer(dim=d_model)
"""
class T5FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_layer1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.gelu = nn.GELU()
        self.dense_layer2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.dense_layer2(self.dropout(self.gelu(self.dense_layer1(hidden_states))))
        return hidden_states
