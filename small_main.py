import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_scheduler
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from small_model import *
import json
import argparse
import time
import os
import wandb #TODO


class KCCdataset(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            self.train = json.load(f)
    def __len__(self):
        return len(self.train)
    def __getitem__(self, index):
        line = self.train[index]
        label = line['label']
        length = len(line['input_ids'])
        return line['input_ids'], label, length


def collate_fn(batch):
    lines, labels, lengths = zip(*batch)
    # max_len = len(max(lines))
    max_len =  max([x for x in lengths if x < 256])
    max_512 = max(lengths)

    ids_res_256, ids_res_512, over_512_len = [], [], []
    for i, line, lens in enumerate(zip(lines, lengths)):
        len_ids = max_len - lens
        len_512 = max_512 - lens
       # if len_ids != 0:
        if lens < 256:
            padding = torch.ones((1, abs(len_ids)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_256.append(ids_tensor)
        else:
            padding = torch.ones((1, abs(len_512)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_512.append(ids_tensor)
            over_512_len.append(i)
        
        # if lens < 256:
        #     ids_res_256.append(ids_tensor)
        # else:
        #     ids_res_512.append(ids_tensor)
        #     over_512_len.append(i)
        # label_res.append(torch.LongTensor([label]))
    ids_batch_256 = torch.cat(ids_res_256, dim=0)
    if ids_res_512 != []:
        ids_batch_512 = torch.cat(ids_res_512, dim=0)
    else:
        ids_batch_512 = torch.LongTensor(ids_res_512)
    
    label_batch = torch.LongTensor(labels).reshape(-1)
    len_batch = torch.LongTensor(lengths)
    return {'256':ids_batch_256, '512':ids_batch_512}, label_batch, len_batch, over_512_len
     

def train(model, dataloader, criterion, optimizer, lr_scheduler, config, train_begin, epoch):
    model.train()
    begin = epoch_begin = time.time()
    cum_loss, cum_acc = 0, 0
    batch, print_batch = 0, 100
    total_num = len(dataloader)
    if config.dataset == 'imdb':
        for inputs_dict, labels, lengths, over_idx in dataloader:
            inputs, labels, lengths = inputs_dict['256'].to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
                
            if inputs_dict['512'].size(0) != 0:
                long_inputs = inputs_dict['512'].to(config.device)
                optimizer.zero_grad()
                outputs = model(long_inputs, lengths[over_idx]) # (B, 2)
                loss = criterion(outputs, labels[over_idx])
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                acc = torch.sum((out_idx == labels[over_idx]) / out_idx.shape[0], dim=0).item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            optimizer.zero_grad()

            ## 512 token idx 선별 후 0으로 만든 뒤 drop
            lengths[over_idx] = 0
            lengths = lengths[lengths != 0]
            labels[over_idx] = 999
            labels = labels[labels != 999]
            outputs = model(inputs, lengths) # (B, 2)
            loss = criterion(outputs, labels)
            out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
            out_idx = torch.max(outputs, 1)[1]
            acc = torch.sum((out_idx == labels) / out_idx.shape[0], dim=0).item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            cum_loss += loss.item()
            cum_acc += acc

            batch += 1
            if batch % print_batch == 0:
                current = time.time()
                elapsed = current - begin
                epoch_elapsed = (current - epoch_begin) / 60.0
                train_elapsed = (current - train_begin) / 3600.0

                print('epoch: {:4d}, batch: {:5d}/{:5d}, \nloss: {:.8f}, elapsed: {:6.2f}s {:6.2f}m {:6.2f}h'.format(
                        epoch, batch, total_num,
                        cum_loss / batch, # moving average loss
                        elapsed, epoch_elapsed, train_elapsed))
                begin = time.time()
    return cum_loss / total_num, cum_acc / batch


def evaluate(model, dataloader, criterion, config):
    model.eval() # Dropout 종료
    
    cum_loss, cum_acc, short_batch = 0, 0, 0 
    cum_long_loss, cum_long_acc, long_leng, batch = 0, 0, 0, 0
    with torch.no_grad():
        for inputs_dict, labels, lengths, over_idx in dataloader:
            inputs, labels, lengths = inputs_dict['256'].to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
            
            if inputs_dict['512'].size(0) != 0:
                long_inputs = inputs_dict['512'].to(config.device)
#                 optimizer.zero_grad()
                outputs = model(long_inputs, lengths[over_idx]) # (B, 2)
                val_loss = criterion(outputs, labels[over_idx])
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                val_acc = torch.sum((out_idx == labels[over_idx]) / out_idx.shape[0], dim=0).item()
                cum_long_loss += val_loss.item()
                cum_long_acc += val_acc
                long_leng += len(over_idx)
                batch += 1
            ## 512 token idx 선별 후 0으로 만든 뒤 drop
            lengths[over_idx] = 0
            lengths = lengths[lengths != 0]
            labels[over_idx] = 999
            labels = labels[labels != 999]
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
            out_idx = torch.max(outputs, 1)[1]
            acc = torch.sum((out_idx == labels) / out_idx.shape[0], dim=0).item()

            cum_loss += loss.item()
            cum_acc += acc
            short_batch += 1
        short_len = len(dataloader) - long_leng
    return cum_loss / short_len, cum_acc / short_batch, cum_long_loss / long_leng, cum_long_acc / batch


def save(filename, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    print('Model saved')


if __name__=='__main__':
    wandb.init(project='kcc', entity='vmtmxmf5') #TODO
    
    parser = argparse.ArgumentParser(description='Transformer-R')
    # model config
    parser.add_argument('-layer', '--num_layers', type=int, default=4, help='# of layers')
    parser.add_argument('-head', '--heads', type=int, default=8, help='# of heads')
    parser.add_argument('--d_model', type=int, default=512, help='d_model')
    parser.add_argument('-ff', '--d_ff', type=int, default=2048, help='feed forward dimensions')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='attn dropout rate')
    parser.add_argument('-pad', '--pad_token_id', type=int, default=0, help='Huggingface pad token id')
    parser.add_argument('-max_len', '--max_position_embeddings', type=int, default=512, help='For ignore idx, add 2 dim')
    parser.add_argument('-class', '--num_labels', type=int, default=2, help='# of classes in labels')
    parser.add_argument('-trim', '--d_trim', type=int, default=512, help='MHA trimmed dimension')
    parser.add_argument('-vocab', '--vocab_size', type=int, default=30522, help='Roberta-large Tokenizer Vocab sizes')
    
    # customizing hyperparms.
    parser.add_argument('--dataset', default='imdb', choices=['sst2', 'qqp', 'qnli', 'enwik8', 'text8', 'imdb'], help='what kind of datasets are you using?')
    parser.add_argument('-pre', '--pretrained_weights', type=str, default='Small.pt', help='modified pretrained weights path')
    
    ## 실험모드 추후 반드시 수정할 것
    parser.add_argument('-B', '--batch_size', type=int, default=128, help='Batch sizes')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('-cuda', '--device', type=str, default='cpu', help='If you want to use cuda, then write down cuda:0 or smt else')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    config = parser.parse_args()
    
    ### dataset settings
    if config.dataset == 'imdb':
        paths = ['imdb_train.json', 'imdb_valid.json']
    elif config.dataset == ('enwik8' or 'text8'):
        ## bpc mode 
        ## dataset 수정 필요!
        paths = [config.dataset + '_' + name + '.txt' for name in ['train', 'valid', 'test']]
    else:
        paths = [config.dataset + '_' + name + '.json' for name in ['train', 'valid', 'test']]
    
    ### device
    if config.device != 'cpu':
        config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    else:
        config.device = torch.device(config.device)


    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_ds = KCCdataset(paths[0])
    valid_ds = KCCdataset(paths[1])
    train_dl = DataLoader(train_ds,
                        config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds,
                        config.batch_size,
                        num_workers=config.num_workers,
                        collate_fn=collate_fn)


    model = ClassificationHead(config).to(config.device)
    pretrained_dict = torch.load(config.pretrained_weights, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict['model'])
    model.load_state_dict(model_dict)
    
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log='all', log_freq=100) #TODO
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_training_steps = config.epochs * len(train_dl)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=8000,
                                num_training_steps=num_training_steps)
    
    train_begin = time.time()
    best_valid_loss = 100
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_dl, criterion, optimizer, lr_scheduler, config, train_begin, epoch)
        wandb.log({'train_loss':train_loss, 'train_acc':train_acc}) #TODO

        short_ppl, short_acc, long_ppl, long_acc = evaluate(model, valid_dl, criterion, config)
        wandb.log({'s1_ppl':torch.exp(short_ppl), 's1_acc':short_acc}) #TODO
        wandb.log({'s2_ppl':torch.exp(long_ppl), 's2_acc':long_acc})
        if short_ppl < best_valid_loss:
            save(os.path.join('checkpoint', f'model_{epoch+1:03d}.pt'), model, optimizer)
            best_valid_loss = short_ppl
