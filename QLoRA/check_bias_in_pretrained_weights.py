import torch
from t5.encoder_decoder import T5Transformer
from t5.config import T5Config

config = T5Config()
model = T5Transformer(config).to(config.device)

check_point = '../t5_pretraining_best_model.pt'
state_dict = torch.load(check_point, map_location=config.device)
model_weights = state_dict['model_state_dict']

model.load_state_dict(model_weights)


for name, _ in model.named_parameters():
    if 'bias' not in name:
        print(name);print('no bias');print()


