import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, config, use_relative_attention_bias=False):
        super().__init__()
        self.d_model = config.d_model # 512
        self.num_heads = config.num_heads # 8
        self.head_dim = self.d_model // self.num_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.attention_weights_dropout_rate)

        self.use_relative_attention_bias = use_relative_attention_bias
        if self.use_relative_attention_bias:
            self.num_buckets = config.num_buckets
            self.max_distance = config.max_distance
            self.relative_attention_bias_lookup_table = nn.Embedding(config.num_buckets, config.num_heads)

    # https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    @staticmethod
    def _get_relative_position_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = max_exact + \
                        (torch.log(relative_position.float() / max_exact)
                        / math.log(max_distance / max_exact)
                        * (num_buckets - max_exact)).to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        relative_buckets += torch.where(is_small, relative_position, val_if_large)
        return relative_buckets

    def _compute_bias(self, query_length, key_length, is_decoder, device):
        """
        calculate relative position bias to add attention score
        """
        context_position = torch.arange(query_length, dtype=torch.long).unsqueeze(-1).to(device) # attending position
        memory_position = torch.arange(key_length, dtype=torch.long).unsqueeze(0).to(device)
        # context_position.shape: [query_length, 1]
        # memory_position.shpae: [1, key_length]

        relative_position = memory_position - context_position
        # broadcasting: [1, key_length] - [query_length, 1]
        # => [query_length, key_length] - [query_length, key_length] = [query_length, key_length]

        #  encoder self-attention == bidirectional # decoder self-attention == not bidirectional
        relative_position_bucket = self._get_relative_position_bucket(relative_position, is_decoder, self.num_buckets, self.max_distance)
        relative_attention_bias = self.relative_attention_bias_lookup_table(relative_position_bucket)
        return relative_attention_bias

    def _split_heads(self, tensor):
        """
        [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_length, _ = tensor.shape
        return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _concat_heads(self, tensor):
        """
        [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_length, head_dim = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads*head_dim)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class EncoderSelfAttention(Attention):
    def __init__(self, config):
        super().__init__(config, use_relative_attention_bias=True)
        self.is_encoder = True

    def forward(self, hidden_states, mask, position_bias):
        query_length, key_length = hidden_states.shape[1], hidden_states.shape[1]
        # projection
        Q, K, V = self.W_q(hidden_states), self.W_k(hidden_states), self.W_v(hidden_states)

        # split heads
        Q_h, K_h, V_h = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        # attention scores
        # q_h.shape: [batch_size, num_heads, (query) seq_length, head_dim]
        # k_h.shpae: [batch_size, num_heads, (key) seq_length, head_dim]
        scores = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)
        # scores.shape: [batch_size, num_heads, (query) seq_length, (key) seq_length]

        # relative_attention_bias
        if position_bias is None:
            position_bias = self._compute_bias(query_length, key_length, self.is_encoder, hidden_states.device)
            position_bias = position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        # relative_attention_bias.shape: [1, num_heads, (query) seq_length, (key) seq_length]

        # attention scores + relative attention bias
        scores += position_bias

        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)

        # attention weights
        weights = self.dropout(self.softmax(scores))

        # attention value(context vector)
        context_h = weights @ V_h

        # concatenate attention heads
        context = self.W_o(self._concat_heads(context_h))
        return context, position_bias

class DecoderSelfAttentionWithKcache(Attention):
    def __init__(self, config):
        super().__init__(config, use_relative_attention_bias=True)
        self.is_encoder = True
        self.register_buffer('k_cache', None, persistent=False)
        self.register_buffer('W_kv', None, persistent=True)

    def precompute_w_kv(self):
        W_k = self.W_k.weight.T.detach()
        W_v = self.W_v.weight.T.detach()

        W_k_inv = torch.linalg.inv(W_k)
        self.W_kv = W_k_inv @ W_v

    def reset_k_cache(self):
        self.k_cache = None

    def _training_forward(self, hidden_states, mask, position_bias):
        query_length, key_length = hidden_states.shape[1], hidden_states.shape[1]

        Q, K, V = self.W_q(hidden_states), self.W_k(hidden_states), self.W_v(hidden_states)
        Q_h, K_h, V_h = self._split_heads(Q), self._split_heads(K), self._split_heads(V)
        scores = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)

        if position_bias is None:
            position_bias = self._compute_bias(query_length, key_length, (not self.is_encoder), hidden_states.device)
            position_bias = position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        scores += position_bias
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)

        weights = self.dropout(self.softmax(scores))
        context_h = weights @ V_h
        context = self.W_o(self._concat_heads(context_h))
        return context, position_bias

    def _inference_forward(self, hidden_states, mask, position_bias):
        is_prompt_phase = self.k_cache is None

        if is_prompt_phase:
            query_length, key_length = hidden_states.shape[1], hidden_states.shape[1]

            Q = self.W_q(hidden_states)
            self.k_cache = self.W_k(hidden_states) # K
            V = self.k_cache @ self.W_kv # V = K(W_k_inv @ W_v) = K·W_kv

            Q_h, K_h, V_h = self._split_heads(Q), self._split_heads(self.k_cache), self._split_heads(V)
            scores = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)

            if position_bias is None:
                """
                in prompt phase, decoder query_length = key_length = 1
                however, generate phase load K-Cache -> query_length = 1, key_length = i (i=1, ... , seq_len=N)
                
                for example, if N = 3 (3 past tokens), the incoming query is the 4th token.
                That is, in the generate phase, relative_position = memory_position - context_position = [[-3, -2, -1, 0]]
                
                therefore, in the generate phase, update K-Cache(self.k_cache), 
                memory_position and context_position using updated K-Cache(exactly, the updated length of self.k_cache)
                """
                position_bias = self._compute_bias(query_length, key_length, (not self.is_encoder), hidden_states.device)
                position_bias = position_bias.permute(2, 0, 1).contiguous().unsqueeze(0) # position_bias.shape: [1, num_heads, 1, 1]

            scores += position_bias
            if mask is not None:
                scores = scores.masked_fill(mask == False, -1e4)

            weights = self.dropout(self.softmax(scores))
            context_h = weights @ V_h
            context = self.W_o(self._concat_heads(context_h))
        else: # generate phase # mask is None
            ## head_i = [softmax((Q_i K_i^T)/sqrt{head_dim}) @ K] @ W_kv, i
            Q_new = self.W_q(hidden_states)
            K_new = self.W_k(hidden_states)
            self.k_cache = torch.concat([self.k_cache, K_new], dim=1) # update K-Cache

            Q_new_h = self._split_heads(Q_new) # [batch_size, num_heads, 1, head_dim] # 1 = query_length
            K_all_h = self._split_heads(self.k_cache) # [batch_size, num_heads, self.k_cache.shape[1]=key_length, head_dim]
            scores = Q_new_h @ K_all_h.transpose(2, 3) / math.sqrt(self.head_dim) # [batch_size, num_heads, 1, key_length] # 1 = query_length

            if position_bias is None:
                context_position = torch.arange(self.k_cache.shape[1], dtype=torch.long).unsqueeze(-1).to(hidden_states.device)
                memory_position = torch.arange(self.k_cache.shape[1], dtype=torch.long).unsqueeze(0).to(hidden_states.device)
                relative_position = memory_position - context_position
                relative_position_bucket = self._get_relative_position_bucket(relative_position, (not self.is_encoder), self.num_buckets, self.max_distance)
                position_bias = self.relative_attention_bias_lookup_table(relative_position_bucket[-1:, :])
                position_bias = position_bias.permute(2, 0, 1).contiguous().unsqueeze(0) # position_bias.shape: [1, num_heads, 1(query_length), key_length]

            scores += position_bias
            if mask is not None:
                scores = scores.masked_fill(mask == False, -1e4)

            weights = self.dropout(self.softmax(scores))
            weights_k = weights @ K_all_h # [batch_size, num_heads, 1, key_length] @ [batch_size, num_heads, key_length, head_dim]
            # [batch_size, num_heads, 1, head_dim]

            context_1 = self._concat_heads(weights_k)  # [batch_size, 1, num_heads * head_dim]

            context_2 = context_1 @ self.W_kv  # [batch_size, 1, d_model] @ [d_model, d_model] -> [batch_size, 1, d_model]

            context = self.W_o(context_2)
        return context, position_bias


    def forward(self, hidden_states, mask, position_bias, use_cache=False):
        if self.training: # model.train()
            return self._training_forward(hidden_states, mask=mask, position_bias=position_bias)
        else: # model.eval()
            # when training, 100% teacher forcing is used.
            # autoregressive generation for evaluation would create a large gap in performance
            # to avoid a large performance gap, teacher forcing is applied when cache is not used.
            if use_cache:
                return self._inference_forward(hidden_states, mask=mask, position_bias=position_bias)
            else:
                return self._training_forward(hidden_states, mask=mask, position_bias=position_bias)


class CrossAttentionWithECache(Attention):
    def __init__(self, config):
        super().__init__(config, use_relative_attention_bias=False)
        self.register_buffer('e_cache_k', None, persistent=False)
        self.register_buffer('e_cache_v', None, persistent=False)

    def reset_e_kv_cache(self):
        self.e_cache_k, self.e_cache_v = None, None

    def forward(self, hidden_states, key_value_states, mask, use_cache=False):
        if self.training: # model.train()
            Q, K, V = self.W_q(hidden_states), self.W_k(key_value_states), self.W_v(key_value_states)

        else: # model.eval()
            if use_cache:
                Q = self.W_q(hidden_states)

                if self.e_cache_k is not None:
                    K, V = self.e_cache_k, self.e_cache_v
                else:
                    K, V = self.W_k(key_value_states), self.W_v(key_value_states)
                    self.e_cache_k, self.e_cache_v = K, V
            else:
                Q, K, V = self.W_q(hidden_states), self.W_k(key_value_states), self.W_v(key_value_states)

        Q_h, K_h, V_h = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        scores = Q_h @ K_h.transpose(2, 3) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e4)

        weights = self.dropout(self.softmax(scores))
        context_h = weights @ V_h
        context = self.W_o(self._concat_heads(context_h))
        return context







