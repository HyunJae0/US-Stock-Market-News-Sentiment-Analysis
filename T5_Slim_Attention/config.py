import torch

class T5Config:
    def __init__(
            self,
            vocab_size = 50359,
            pad_idx=50258,
            d_model=512,
            d_ff=2048,
            num_heads=8,
            num_layers=6,
            attention_weights_dropout_rate=0.1,
            dropout_rate=0.1,
            layer_norm_eps=1e-6,

            num_buckets=32,
            max_distance=128,

            batch_size=64,
            epochs=40,
            learning_rate=0.01,
            warmup_steps=6680, # num of samples in train datasets=106,917 & drop last batch => steps per epoch=1670
            # epochs=40 -> total train steps=1670*40=66800 # warmup ratio=10% -> warmup_steps = 6680
            clip=0.1,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_weights_dropout_rate = attention_weights_dropout_rate
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps

        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.clip=clip
        self.device = device
