import torch
import torch.nn.functional as F
import numpy as np
import random

def random_segmentation(num_corrupted_tokens, num_span):
    # possible divider candidates
    possible_divider_indices = range(1, num_corrupted_tokens)

    # sampling divider_indices
    divider_indices = sorted(random.sample(possible_divider_indices, num_span-1))

    # calculating the length(# of tokens) of each span using numpy.diff
    boundaries = np.array([0] + divider_indices + [num_corrupted_tokens])
    span_lengths = np.diff(boundaries).tolist()
    return span_lengths

def create_span_corruption_mask(seq_length, corruption_rate, mean_span_length, device):
    num_corrupted_tokens = int(round(seq_length * corruption_rate))
    num_span = int(round(num_corrupted_tokens / mean_span_length))

    num_non_corrupted_tokens = seq_length - num_corrupted_tokens

    noise_span_lengths = random_segmentation(num_corrupted_tokens, num_span)
    non_noise_span_lengths = random_segmentation(num_non_corrupted_tokens, num_span)

    noise_mask = torch.zeros(seq_length).to(bool).to(device)
    current_position = 0

    for i in range(num_span):
        non_noise_length = non_noise_span_lengths[i]
        current_position += non_noise_length

        if current_position < seq_length:
            noise_length = noise_span_lengths[i]
            end_position = min(current_position + noise_length, seq_length)
            noise_mask[current_position:end_position] = True
            current_position += noise_length
    return noise_mask

def convert_t5_input_format(input_token_ids, noise_mask):
    prev_token_is_noise = F.pad(noise_mask[:-1], (1, 0))

    is_first_part_noise_tokens = torch.logical_and(noise_mask, ~prev_token_is_noise)
    is_subsequent_noise_tokens = torch.logical_and(noise_mask, prev_token_is_noise)

    sentinel_ids = 50359 - torch.cumsum(is_first_part_noise_tokens, dim=0, dtype=torch.long) # '<extra_id_0>' token_id: 50259 '<extra_id_99>' token_id: 50358
    # 50359(vocab_size) - 1(first sentinel token) = 50358 -> vocab['<extra_id_99>']
    # 50359(vocab_size) - 2(second sentinel token) = 50357 -> vocab['<extra_id_98>']
    # ...

    updated_tokens = torch.where(is_first_part_noise_tokens, sentinel_ids, input_token_ids)
    input_sequence = updated_tokens[~is_subsequent_noise_tokens]
    return input_sequence

def convert_t5_target_format(input_token_ids, noise_mask):
    target_sequence = convert_t5_input_format(input_token_ids, ~noise_mask)
    return target_sequence
