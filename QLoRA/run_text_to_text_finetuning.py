import torch
import torch.nn as nn
import bitsandbytes as bnb

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm

from t5.encoder_decoder import T5Transformer
from t5.config import T5Config
from qlora.lora_4bit import LoRALayer4bit, apply_qlora_to_model


def train(model, dataloader, criterion, optimizer, scheduler, tokenizer, config):
    model.train()
    total_loss, num_correct, total_acc = 0, 0, 0

    for batch in tqdm(dataloader, desc='Training ...'):
        input_ids, labels = batch['text'].to(config.device), batch['labels'].to(config.device)

        optimizer.zero_grad()

        logits = model(input_ids, labels)
        loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_preds = tokenizer.batch_decode(predicted_ids.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            if pred.strip() == label.strip(): # decoded_preds and decoded_labels is lower
                num_correct += 1

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc += num_correct / labels.shape[0]
    return total_loss / len(dataloader), total_acc / len(dataloader)

def eval(model, dataloader, criterion, tokenizer, config):
    model.eval()
    total_loss, num_correct, total_acc = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating ...'):
            input_ids, labels = batch['text'].to(config.device), batch['labels'].to(config.device)

            logits = model(input_ids, labels)
            loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))

            predicted_ids = torch.argmax(logits, dim=-1)
            decoded_preds = tokenizer.batch_decode(predicted_ids.tolist(), skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

            for pred, label in zip(decoded_preds, decoded_labels):
                if pred.strip() == label.strip(): # decoded_preds and decoded_labels is lower
                    num_correct += 1

            total_loss += loss.item()
            total_acc += num_correct / labels.shape[0]
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
    model = T5Transformer(config).to(config.device)

    check_point = '../t5_pretraining_best_model.pt'
    state_dict = torch.load(check_point, map_location=config.device)
    model_weights = state_dict['model_state_dict']

    model.load_state_dict(model_weights)
    apply_qlora_to_model(model, r=8, alpha=8, lora_dropout=0, device=config.device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    optimizer = create_paged_optimizer(model, lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-06)
    # epochs: 30, num of train samples: 17,522 batch size: 256, drop last batch: True, warmup ratio: 10%
    # drop last batch & steps per epoch: 17522 / 256 = 68
    # total train steps = epochs * 68 = 2,040 steps
    # warmup steps = 2,040 * 10%
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=2040)

    tokenizer = AutoTokenizer.from_pretrained('../my_gpt2_tokenizer')
    ds = load_dataset('hyunjaehyun/token_ids_dataset_for_t5_finetuning')
    train_ds, valid_ds, test_ds = ds['train'], ds['valid'], ds['test']

    format = {'type': 'torch', 'format_kwargs': {'dtype': torch.long}}
    train_ds.set_format(**format)
    valid_ds.set_format(**format)
    test_ds.set_format(**format)

    train_dataloader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=256, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=True)

    best_acc = 0
    patience_check, patience_limit = 0, 5
    for epoch in range(10):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, scheduler, tokenizer, config)
        print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}')
        valid_loss, valid_acc = eval(model, valid_dataloader, criterion, tokenizer, config)
        print(f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.2f}')

        if valid_acc < best_acc:
            patience_check += 1

            if patience_check >= patience_limit: break

        else:
            best_acc = valid_loss
            patience_check = 0
            torch.save(model.state_dict(), 'best_finetuning_model.pt')
            print(f'New model saved with validation accuracy: {best_acc:.2f}')