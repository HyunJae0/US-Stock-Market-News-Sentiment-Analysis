import torch
import torch.nn as nn
import torch.nn.init as init
import bitsandbytes as bnb

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm

from t5.encoder_decoder import T5Transformer
from t5.config import T5Config


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, 3, dtype=torch.bfloat16) # num_labels=3

        init.trunc_normal_(self.dense.weight, std=0.02)
        init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, hiiden_states):
        hidden_states = self.dropout(hiiden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.classifier(hidden_states)
        return hidden_states

class SentimentClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = T5Transformer(config)
        self.classification_head  = ClassificationHead(config)

    def forward(self, input_ids, decoder_input_ids):
        if decoder_input_ids is None:
            decoder_start_token_id = 50257
            shifted_input_ids = input_ids.new_zeros(input_ids.shape) # shape: (batch_size, seq_length)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id # first token of shift_input_ids is <sos>

            decoder_input_ids = shifted_input_ids # <sos> input_token_ids <pad>

        src_mask = self.transformer.create_encoder_mask(input_ids)
        trg_mask = self.transformer.create_decoder_mask(decoder_input_ids)

        # in this task, pretrained model's lm_head is not required
        # so, run only the necessary components instead of the full T5Transformer.forward()
        encoder_output = self.transformer.Encoder(input_ids, src_mask)
        output = self.transformer.Decoder(decoder_input_ids, encoder_output, trg_mask, src_mask)

        # use the first token for classification, as in BERT <cls> token
        sentence_representation = output[:, 0, :] # shape: (batch_size, d_model)
        logits = self.classification_head(sentence_representation)
        # logits.shape: (batch_size, 3(num_labels))
        return logits


def train(model, dataloader, criterion, optimizer, scheduler, config):
    model.train()
    total_loss, total_acc = 0, 0

    for batch in tqdm(dataloader, desc='Training ...'):
        input_ids, labels = batch['input_ids'].to(config.device), batch['labels'].to(config.device)

        optimizer.zero_grad()

        logits = model(input_ids, None)
        loss = criterion(logits.view(-1, 3), labels.view(-1))
        print(logits.shape)

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).sum().item() / labels.shape[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += acc
    return total_loss / len(dataloader), total_acc / len(dataloader)

def eval(model, dataloader, criterion, config):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating ...'):
            input_ids, labels = batch['input_ids'].to(config.device), batch['labels'].to(config.device)

            logits = model(input_ids, None)
            loss = criterion(logits.view(-1, 3), labels.view(-1))
            print(logits.shape)

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).sum().item() / labels.shape[0]

            total_loss += loss.item()
            total_acc += acc
        return total_loss / len(dataloader), total_acc / len(dataloader)

def create_paged_optimizer(model, lr, weight_decay, betas, eps):
    trainable_params = []
    for name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
            trainable_params.append(params)

    kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
        'eps':eps,
        'betas': betas,
    }

    optimizer = bnb.optim.PagedAdamW(trainable_params, **kwargs)
    return optimizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = T5Config()
    model = SentimentClassification(config).to(config.device)

    check_point = 'last_epoch_model.pt'
    state_dict = torch.load(check_point, map_location=config.device, weights_only=False)
    model_weights = state_dict['model_state_dict']

    model.transformer.load_state_dict(model_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    optimizer = create_paged_optimizer(model, lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-06)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=800, num_training_steps=8190)

    tokenizer = AutoTokenizer.from_pretrained('my_gpt2_tokenizer')
    ds = load_dataset('hyunjaehyun/token_ids_dataset_for_t5_finetuning2')
    train_ds, valid_ds, test_ds = ds['train'], ds['valid'], ds['test']

    format = {'type': 'torch', 'format_kwargs': {'dtype': torch.long}}
    train_ds.set_format(**format)
    valid_ds.set_format(**format)
    test_ds.set_format(**format)

    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=64, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=True)

    best_acc = 0
    patience_check, patience_limit = 0, 5
    for epoch in range(30):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, scheduler, config)
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}')
        valid_loss, valid_acc = eval(model, valid_dataloader, criterion, config)
        print(f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.2f}')

        if valid_acc > best_acc:
            best_acc = valid_acc
            patience_check = 0
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_acc,
                'acc': valid_acc,
                'scheduler_state_dict': scheduler.state_dict()},'best_finetuning_model.pt')
            print(f'New model saved with validation accuracy: {best_acc:.2f}')
        else:
            patience_check += 1

            if patience_check == patience_limit: break
