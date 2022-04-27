# longformer
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_scheduler
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from roberta import *
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
        ### line : {'input_ids':[], 'attention_mask':[], 'label':0 or 1}
        line = self.train[index]
        label = line['label']
        length = len(line['input_ids'])
        return line['input_ids'], label, length, line['attention_mask']


def collate_fn(batch):
    lines, labels, lengths, attn = zip(*batch)
    # max_len = len(max(lines))
    max_len = max(lengths)
    
    ids_res, attn_res = [], []
    ### padding을 만든다
    for line, att, lens in zip(lines, attn, lengths):
        ### batch 안에 있는 문장중 가장 긴 문장 길이 - 현재 문장 길이 만큼 pad를 넣는다
        len_ids = max_len - lens
        if len_ids != 0:
            padding = torch.ones((1, len_ids), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            att_tensor = torch.cat([torch.LongTensor([att]), padding], dim=1)
        ### batch 안에 max length 문장의 경우 굳이 pad를 할 필요가 없다 (그 자체로 가장 긴 문장이므로)
        else:
            ids_tensor = torch.LongTensor([line])
            att_tensor = torch.LongTensor([att])
        ids_res.append(ids_tensor)
        attn_res.append(att_tensor)
        # label_res.append(torch.LongTensor([label]))
    ids_batch = torch.cat(ids_res, dim=0)
    attn_batch = torch.cat(attn_res, dim=0)
    label_batch = torch.LongTensor(labels).reshape(-1)
    len_batch = torch.LongTensor(lengths)
    return {'input_ids':ids_batch, 'attention_mask':attn_batch}, label_batch, len_batch
     

def train(model, dataloader, criterion, optimizer, lr_scheduler, config, train_begin, epoch):
    model.train()
    begin = epoch_begin = time.time()
    cum_loss, cum_acc = 0, 0
    batch, print_batch = 0, 100
    total_num = len(dataloader)
    if config.dataset == 'imdb':
        for inputs, labels, lengths in dataloader:
            ### 문장길이 512 이하만 받아온다
            if inputs['input_ids'].shape[1] < 513:
                ### 데이터를 gpu에 올린다
                inputs = {k:v.to(config.device) for k, v in inputs.items()}
                labels, lengths = labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
                optimizer.zero_grad()
                
                ### output을 loss에 넣는다.
                ### loss가 cross entropy이므로, 함수 안에 softmax가 내장되어 있다
                outputs = model(inputs) # (B, 2)
                loss = criterion(outputs, labels)
                
                ### acc를 계산하기 위해 logit 값에 softmax를 취한다
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                ### softmax 중 가장 큰 값을 뽑아 그 인덱스를 반환한다
                out_idx = torch.max(outputs, 1)[1]
                ### label과 예측 index와 같은 개수 / batch
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
    
    cum_loss, cum_acc, batch = 0, 0, 0
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            if inputs['input_ids'].shape[1] < 513:
                inputs = {k:v.to(config.device) for k, v in inputs.items()}
                labels, lengths = labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                acc = torch.sum((out_idx == labels) / out_idx.shape[0], dim=0).item()

                cum_loss += loss.item()
                cum_acc += acc
                batch += 1
    return cum_loss / len(dataloader), cum_acc / batch


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
    parser.add_argument('-layer', '--num_layers', type=int, default=24, help='# of layers')
    parser.add_argument('-head', '--heads', type=int, default=16, help='# of heads')
    parser.add_argument('--d_model', type=int, default=1024, help='d_model')
    parser.add_argument('-ff', '--d_ff', type=int, default=4096, help='feed forward dimensions')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='attn dropout rate')
    parser.add_argument('-pad', '--pad_token_id', type=int, default=1, help='Huggingface pad token id')
    parser.add_argument('-max_len', '--max_position_embeddings', type=int, default=4096+2, help='For ignore idx, add 2 dim')
    parser.add_argument('-class', '--num_labels', type=int, default=2, help='# of classes in labels')
    parser.add_argument('-trim', '--d_trim', type=int, default=256, help='MHA trimmed dimension')
    parser.add_argument('-vocab', '--vocab_size', type=int, default=50265, help='Roberta-large Tokenizer Vocab sizes')
    
    # customizing hyperparms.
    parser.add_argument('--dataset', default='imdb', choices=['sst2', 'qqp', 'qnli', 'enwik8', 'text8', 'imdb'], help='what kind of datasets are you using?')
    parser.add_argument('-pre', '--pretrained_weights', type=str, default='Roberta.pt', help='modified pretrained weights path')
    
    ## 실험모드 추후 반드시 수정할 것
    parser.add_argument('-B', '--batch_size', type=int, default=3, help='Batch sizes')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('-cuda', '--device', type=str, default='cpu', help='If you want to use cuda, then write down cuda:0 or smt else')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate')
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


    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
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


    model = LongformerHead(config).to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log='all', log_freq=100) #TODO
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_training_steps = config.epochs * len(train_dl)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=8000,
                                num_training_steps=num_training_steps)
    
    train_begin = time.time()
    best_valid_loss = 100.0
    for epoch in range(config.epochs):
        train_loss, train_acc = train(model, train_dl, criterion, optimizer, lr_scheduler, config, train_begin, epoch)
        wandb.log({'train_loss':train_loss, 'train_acc':train_acc}) #TODO

        valid_loss, valid_acc = evaluate(model, valid_dl, criterion, config)
        wandb.log({'valid_loss':valid_loss, 'valid_acc':valid_acc}) #TODO
        if valid_loss < best_valid_loss:
            save(os.path.join('checkpoint', f'model_{epoch+1:03d}.pt'), model, optimizer)
            best_valid_loss = valid_loss
