from t5.layernorm import T5LayerNorm
from t5.feed_forward_network import T5FeedForwardNetwork
from t5.slim_attention_and_relative_position_bias import *

class T5EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. self-attention
        self.layer_norm_1 = T5LayerNorm(config)
        self.encoder_self_attention = EncoderSelfAttention(config) # is_encoder=True
        self.dropout_1 = nn.Dropout(config.dropout_rate)

        # 2. feed forward network
        self.layer_norm_2 = T5LayerNorm(config)
        self.feed_forward_network = T5FeedForwardNetwork(config)
        self.dropout_2 = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, mask, position_bias):
        # Pre-LN # LayerNorm -> Attention -> Dropout
        _hidden_states = hidden_states
        norm1 = self.layer_norm_1(_hidden_states)
        attn_output, position_bias = self.encoder_self_attention(norm1, mask, position_bias)
        hidden_states = _hidden_states + self.dropout_1(attn_output) # add (residual Connection)

        _hidden_states = hidden_states
        norm2 = self.layer_norm_2(_hidden_states)

        ffn_output = self.feed_forward_network(norm2)
        hidden_states = _hidden_states + self.dropout_2(ffn_output)
        return hidden_states, position_bias # share position_bias across all layers

class T5Encoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.encoder_layers = nn.ModuleList([T5EncoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, src, src_mask):
        hidden_states = self.embedding(src)

        # initially pass None to compute the relative position bias in the first encoder/decoder layer,
        # then reuse it for all remaining encoder/decoder layers.
        position_bias = None
        for encoder_layer in self.encoder_layers:
            hidden_states, position_bias = encoder_layer(hidden_states, src_mask, position_bias)
        return hidden_states

class T5DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. self-attention
        self.layer_norm_1 = T5LayerNorm(config)
        self.decoder_self_attention = DecoderSelfAttentionWithKcache(config)
        self.dropout_1 = nn.Dropout(config.dropout_rate)

        # 2. cross-attention
        self.layer_norm_2 = T5LayerNorm(config)
        self.decoder_cross_attention = CrossAttentionWithECache(config)
        self.dropout_2 = nn.Dropout(config.dropout_rate)

        # 3. feed forward network
        self.layer_norm_3 = T5LayerNorm(config)
        self.feed_forward_network = T5FeedForwardNetwork(config)
        self.dropout_3 = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, encoder_hidden_states, position_bias, trg_mask, src_mask, use_cache=False):
        _hidden_states = hidden_states
        norm1 = self.layer_norm_1(hidden_states)
        attn_output, position_bias = self.decoder_self_attention(norm1, trg_mask, position_bias, use_cache=use_cache)
        hidden_states = _hidden_states + self.dropout_1(attn_output)

        _hidden_states = hidden_states
        norm2 = self.layer_norm_2(hidden_states)
        cross_attn_output = self.decoder_cross_attention(norm2, encoder_hidden_states, src_mask, use_cache=use_cache)
        hidden_states = _hidden_states + self.dropout_2(cross_attn_output)

        _hidden_states = hidden_states
        norm3 = self.layer_norm_3(_hidden_states)
        ffn_output = self.feed_forward_network(norm3)
        hidden_states = _hidden_states + self.dropout_3(ffn_output)
        return hidden_states, position_bias

class T5Decoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.embedding = shared_embedding
        self.decoder_layers = nn.ModuleList([T5DecoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, trg, encoder_output, trg_mask, src_mask, use_cache=False):
        hidden_states = self.embedding(trg)

        position_bias = None
        for decoder_layer in self.decoder_layers:
            hidden_states, position_bias = decoder_layer(hidden_states, encoder_output, position_bias,
                                                             trg_mask, src_mask, use_cache=use_cache)
        return hidden_states

    def reset_all_caches(self):
        for decoder_layer in self.decoder_layers:
            # DecoderSelfAttentionWithKCache cache reset
            decoder_layer.decoder_self_attention.reset_k_cache()
            # CrossAttentionWithEcache cache reset
            decoder_layer.decoder_cross_attention.reset_e_kv_cache()

class T5Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pad_idx = config.pad_idx

        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.Encoder = T5Encoder(config, shared_embedding=self.shared_embedding)
        self.Decoder = T5Decoder(config, shared_embedding=self.shared_embedding)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight

        self._W_kv_precomputed = False

    def create_encoder_mask(self, src):  # padding mask
        encoder_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2).to(src.device)
        return encoder_mask

    def create_decoder_mask(self, trg): # padding mask + causal mask
        batch_size, trg_length = trg.shape
        decoder_mask_1 = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        decoder_mask_2 = torch.tril(torch.ones((trg_length, trg_length), device=trg.device)).bool()
        decoder_mask_2 = decoder_mask_2.unsqueeze(0).unsqueeze(1)
        # decoder_mask_1.shape: [batch_size, 1, 1, trg_length]
        # decoder_mask_2.shape: [1, 1, trg_length, trg_length]
        decoder_mask = decoder_mask_1 & decoder_mask_2
        # decoder_mask.shape: [batch_size, 1, trg_length, trg_length]
        return decoder_mask

    def forward(self, src, trg):
        src_mask = self.create_encoder_mask(src)
        trg_mask = self.create_decoder_mask(trg)

        encoder_output = self.Encoder(src, src_mask)
        decoder_output = self.Decoder(trg, encoder_output, trg_mask, src_mask)

        logits = self.lm_head(decoder_output)
        return logits

    def precompute_all_w_kv(self):
        for decoder_layer in self.Decoder.decoder_layers:
            decoder_layer.decoder_self_attention.precompute_w_kv()
        self._W_kv_precomputed = True

    @torch.no_grad()
    def generate(self, src, max_len=512, sos_token_id=50257, eos_token_id=50256, pad_token_id=50258):
        self.eval()

        self.Decoder.reset_all_caches()

        if not self._W_kv_precomputed:
            self.precompute_all_w_kv() # Precompute the W_kv matrices for all decoder layers

        batch_size = src.shape[0]

        ## prompt phase
        src_mask = self.create_encoder_mask(src)
        encoder_stack_output = self.Encoder(src, src_mask) # E-cache

        # tensor to store the generated tokens
        trg_tokens = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=src.device)

        ## generate phase # autoregressive greedy decoding
        generation_finish_mask = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len-1):
            current_token = trg_tokens[:, -1:]

            decoder_output = self.Decoder(
                trg=current_token,
                encoder_output=encoder_stack_output,
                trg_mask=None,
                src_mask=src_mask,
                use_cache=True
            )
            # decoder_output.shape: [batch_size, seq_length=1, d_model]
            logits = self.lm_head(decoder_output.squeeze(1)) # [batch_size, vocab_size]

            # greedy decoding
            next_token = torch.argmax(logits, dim=1) # next_token.shape: [batch_size]

            # torch.where(condition, A, B)
            # if condition is True -> A # condition is False -> B
            # assign eos_token_id for finished sequences and next_token for ongoing sequences based on generation_finish_mask
            next_token = torch.where(generation_finish_mask,
                                     torch.tensor(pad_token_id, device=src.device), next_token)

            # generation_finish_mask update
            generation_finish_mask |= (next_token == eos_token_id)

            next_token = next_token.unsqueeze(1) # next_token.shape: [batch_size, 1]
            trg_tokens = torch.cat([trg_tokens, next_token], dim=1)

            if generation_finish_mask.all(): break # terminate generate phase when all sequences generated
        return trg_tokens