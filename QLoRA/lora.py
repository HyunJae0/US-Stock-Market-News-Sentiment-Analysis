#  Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class LoRA:
    def __init__(
            self,
            r,
            alpha,
            lora_dropout
    ):
        self.r = r
        self.alpha = alpha
        # Optional dropout
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

class LoRALayer(nn.Linear, LoRA):
    def __init__(
            self,
            r,
            alpha,
            lora_dropout,
            in_features, # nn.Linear
            out_features, # nn.Linear
            bias=False, # nn.Linear # my pretrained model has no bias
            **kwargs # nn.Linear
    ):
        LoRA.__init__(self, r, alpha, lora_dropout)
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)

        if r > 0:
            # freeze pre-training model weight matrix
            self.weight.requires_grad = False
            self.scaling = self.alpha / self.r
            """
            when using 'tensor.new_zeros()', if dtype=None(default), device=None(default)
            it has same dtype and device as this tensor 
            """
            self.lora_A = nn.Parameter(self.weight.new_zeros((in_features, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, out_features)))
            """
            given input of downstream task 'x', output 'h'
            
            vanilla fine-tuning: h = x·W₀
            (where W₀ is the pre-trained weight matrix and freezing; "requires_grad = False")
            
            in LoRA paper, output h becomes -> h = W₀·x + ΔW·x = W₀·x + BA·x (3)
            figure. 1 is an expression of (3)
            
            1. x -> W₀ => x·W₀
            2. x -> A => x·A -> B => x·A·B
            3. h = x·W₀ + x·A·B
    
            therefore, lora weight matrix(lora_A, lora_B) shape: (in_features, r), (r, out_features)
            """
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        init.zeros_(self.lora_B)  # B is initialized to 0

    def _get_delta_weight(self):
        return (self.lora_A @ self.lora_B) * self.scaling # (in_features, out_features)

    """
    weight merging -> optional
    
    training phase, there are two separate weight paths(W₀·x, BA·x)
    if x.shape is (batch_size, in_features) -> two matmul operations
    
    however, inference phase does not require weight updates, so two weight paths = inefficient
    this problem can be solved through "weight merging"
    
    weight merging is optional, only one matmul operation occurs
    because of W₀.shape: (in_features, out_features) and ΔW.shape: (in_features, out_features)
    using distributive property of matrix 
    W₀·x + ΔW·x = (W₀ + ΔW) ·x
    """
    def train(self, mode=True):
        nn.Linear.train(self, mode)

        delta_w = self._get_delta_weight()
        if mode: # train
            if self.merged:
                self.weight.data -= delta_w.T # un-merge
            self.merged = False
        else:
            if not self.merged:
                self.weight.data += delta_w.T # re-merge
            self.merged = True
        # self.weight(=nn.Linear(in_features, out_features).weight).shape: (in_features, out_features)^T

    def forward(self, x):
        if self.r > 0 and self.merged:
            output = F.linear(x, self.weight, bias=None) # x·W₀
            # x.shape: (batch_size, in_features)
            # self.weight.shape: (in_features, out_features)^T
            # F.linear(x, self.weight, bias=None)
            # => (batch_size, in_features) @ ((in_features, out_features)^T )^T = (batch_size, out_features)
            output += self.lora_dropout(x) @ self._get_delta_weight() # x·W₀ + x·ΔW
            return output
        else:
            return F.linear(x, self.weight, bias=None)


#    def _merge_weights(self):
#        delta_w = self._get_delta_weight()
#        return self.weight.data + delta_w.T

#    def forward(self, x):
        ## torch.nn.functional.linear(input: x, weight: A, bias=None)
        ## x·A^T + b
#        print(self.weight.requires_grad) ## False
#        print(self.weight.shape)  ## shape: (10, 6)
#        output = F.linear(x, self.weight, bias=None)
#        delta_w = self._get_delta_weight()
#        print(delta_w.shape)  ## shape: (6, 10)
        ## x.shape: (4, 6) nn.Linear(lora_layer).shape: (in_features=6, out_features=6)
#        print(output.shape)  ## (4, 6) @ (10, 6)^T -> (4, 10)

#        lora_output = output + self.lora_dropout(x) @ delta_w
#        print(lora_output.shape) ## (4, 6) @ (6, 10) -> (4, 10)
#        lora_output2 = x.detach() @ self._merge_weights()
#        print(lora_output == lora_output2)
#        return lora_output, lora_output2

"""
Apply LoRA to the embedding layer as well 
"""
class LoRAEmbeddingLayer(nn.Embedding, LoRA):
    def __init__(
            self,
            r,
            alpha,
            num_embeddings, # nn.Embedding
            embedding_dim, # nn.Embedding
            lora_dropout,
            **kwargs # nn.Embedding
    ):
        LoRA.__init__(self, r, alpha, lora_dropout=0)
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)

        if r > 0:
            # freeze pre-training model weight matrix
            self.weight.requires_grad = False
            self.scaling = self.alpha / self.r
            self.lora_A = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, embedding_dim)))

        init.zeros_(self.lora_A)
        init.normal_(self.lora_B)

    def _get_delta_weight(self):
        return (self.lora_A @ self.lora_B) * self.scaling # (num_embeddings, embedding_dim)

    def train(self, mode=True):
        nn.Embedding.train(self, mode)

        delta_w = self._get_delta_weight() #
        if mode:
            if self.merged:
                # self.weight(=nn.Embedding(num_embeddings, embedding_dim).weight).shape: (num_embeddings, embedding_dim)
                self.weight.data -= delta_w
            self.merged = False
        else:
            if not self.merged:
                self.weight.data += delta_w
            self.merged = True

    def forward(self, x):
        if self.r > 0 and self.merged:
            orig_layer_output = F.embedding(input=x,
                                            weight=self.weight,
                                            padding_idx=self.padding_idx,
                                            max_norm=self.max_norm,
                                            norm_type=self.norm_type,
                                            scale_grad_by_freq=self.scale_grad_by_freq,
                                            sparse=self.sparse) # pretrained model embedding layer output path # # x·W₀
            # if x.shape: (batch_size, seq_len)
            # orig_layer_output.shape: (batch_size, x seq_len, embedding_dim)
            lora_A_output = F.embedding(input=x,
                                        weight=self.lora_A,
                                        max_norm=self.max_norm,
                                        norm_type=self.norm_type,
                                        scale_grad_by_freq=self.scale_grad_by_freq,
                                        sparse=self.sparse)
            # lora_A_output.shape: (batch_size, x seq_len, embedding_dim)
            lora_output = (lora_A_output @ self.lora_B) * self.scaling
            output = orig_layer_output + lora_output
            return output
        else:
            return F.embedding(input=x, weight=self.weight, padding_idx=self.padding_idx, max_norm=self.max_norm,
                               norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)

#if __name__ == '__main__':
#    lora_layer = LoRALayer(r=2, alpha=1, lora_dropout=0,
#                      in_features=6, out_features=10)
                      # nn.Linear(6, 10)

#    print('1',isinstance(lora_layer, nn.Linear)) # True
#    print('2',lora_layer.weight.requires_grad) # True
#    print('3',lora_layer.bias) # None

#    x = torch.arange(5).unsqueeze(0) # batch_size, in_dim
#    lora_layer(x)

#    output = lora_layer(x)
#    print('4',output)
#    lora_embedding = LoRAEmbedding(r=2, alpha=1, num_embeddings=10, embedding_dim=10)
#    output = lora_embedding(x)

#    print('5', output)
