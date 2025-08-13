import torch
import torch.nn as nn
import torch.nn.init as init
import bitsandbytes as bnb
import math

class LoRALayer4bit(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, lora_dropout, device):
        super().__init__()
        self.layer_4bit = bnb.nn.Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=False,
            compute_dtype=torch.bfloat16,
            device=device
        )

        self.in_features = in_features
        self.out_features = out_features

        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r

        self.lora_A = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16).to(device)
        self.lora_B = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16).to(device)

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        init.zeros_(self.lora_B.weight)

    def _get_delta_weight(self, x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

    def _checkpointed_forward(self, x):
        output = self.layer_4bit(x)
        lora_output = self._get_delta_weight(x)
        return output + lora_output

    def forward(self, x): # x.dtype: torch.bfloat16
        output = self.layer_4bit(x.to(torch.bfloat16))
        if self.r > 0:
            output += self._get_delta_weight(x.to(torch.bfloat16))
            return output
        else:
            return output


def apply_qlora_to_model(model, r, alpha, lora_dropout, device):
    torch_nn_linear_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' not in name and 'classifier' not in name:
            torch_nn_linear_list.append(name)

    for name in torch_nn_linear_list:
        path_parts = name.split('.')
        parent_name = '.'.join(path_parts[:-1])
        child_name = path_parts[-1]

        parent_module = model.get_submodule(parent_name)

        orig_layer = getattr(parent_module, child_name)
        in_features = orig_layer.in_features
        out_features = orig_layer.out_features
        pretrained_weight = orig_layer.weight.data.clone()

        lora_layer = LoRALayer4bit(in_features, out_features, r, alpha, lora_dropout, device)

        quantized_weight, quant_state = bnb.functional.quantize_4bit(
            pretrained_weight,
            blocksize=64,
            compress_statistics=True,
            quant_type='nf4'
        )

        new_weight_param = bnb.nn.Params4bit(
            data=quantized_weight,
            requires_grad=False,
            quant_state=quant_state,
            blocksize=64,
            compress_statistics=True,
            quant_type='nf4'
        )

        lora_layer.layer_4bit.weight = new_weight_param
        setattr(parent_module, child_name, lora_layer)