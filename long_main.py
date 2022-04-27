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
        line = self.train[index]
        label = str(int(line['label']) + 2)
        length = len(line['input_ids'])
        return line['input_ids'], label, length


def collate_fn(batch):
    lines, labels, lengths = zip(*batch)
    # max_len = len(max(lines))
    max_512 =  max([x for x in lengths if x <= 512])
    max_1024 = max([x for x in lengths if 512 < x <= 1024])
    max_2048 = max([x for x in lengths if 1024 < x <= 2048])
    max_4096 = max(lengths)

    ids_res_512, ids_res_1024, over_idx_1024, ids_res_2048, over_idx_2048, ids_res_4096, over_idx_4096 = [], [], [], [], [], [], []
    for i, line, lens in enumerate(zip(lines, lengths)):
        len_512 = max_512 - lens
        len_1024 = max_1024 - lens
        len_2048 = max_2048 - lens
        len_4096 = max_4096 - lens
       # if len_ids != 0:
        if lens <= 512:
            padding = torch.ones((1, abs(len_512)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_512.append(ids_tensor)
        elif 512 <= lens <= 1024:
            padding = torch.ones((1, abs(len_1024)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_1024.append(ids_tensor)
            over_idx_1024.append(i)
        elif 1024 <= lens <= 2048:
            padding = torch.ones((1, abs(len_2048)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_2048.append(ids_tensor)
            over_idx_2048.append(i)
        else:
            padding = torch.ones((1, abs(len_4096)), dtype=torch.long)
            ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
            ids_res_4096.append(ids_tensor)
            over_idx_4096.append(i)
        
    ids_batch_512 = torch.cat(ids_res_512, dim=0)
    if ids_res_1024 != []:
        ids_batch_1024 = torch.cat(ids_res_1024, dim=0)
    else:
        ids_batch_1024 = torch.LongTensor(ids_res_1024)
    
    if ids_res_2048 != []:
        ids_batch_2048 = torch.cat(ids_res_2048, dim=0)
    else:
        ids_batch_2048 = torch.LongTensor(ids_res_2048)
    
    if ids_res_4096 != []:
        ids_batch_4096 = torch.cat(ids_res_4096, dim=0)
    else:
        ids_batch_4096 = torch.LongTensor(ids_res_4096)
    
    label_batch = torch.LongTensor(labels).reshape(-1)
    len_batch = torch.LongTensor(lengths)
    return {'512':ids_batch_512, '1024':ids_batch_1024, '2048':ids_batch_2048, '4096':ids_batch_4096}, label_batch, len_batch, over_idx_1024, over_idx_2048, over_idx_4096
     

def train(model, dataloader, criterion, optimizer, lr_scheduler, config, train_begin, epoch):
    model.train()
    begin = epoch_begin = time.time()
    cum_loss, cum_acc = 0, 0
    batch, print_batch = 0, 100
    total_num = len(dataloader)
    for inputs_dict, labels, lengths, over_idx_1024, over_idx_2048, over_idx_4096 in dataloader:
        inputs, labels, lengths = inputs_dict['512'].to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
                
        if inputs_dict['1024'].size(0) != 0:
            long_inputs = inputs_dict['1024'].to(config.device)
            optimizer.zero_grad()
            outputs = model(long_inputs, lengths[over_idx_1024]) # (B, 2)
            loss = criterion(outputs, labels[over_idx_1024])
            out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
            out_idx = torch.max(outputs, 1)[1]
            acc = torch.sum((out_idx == labels[over_idx_1024]) / out_idx.shape[0], dim=0).item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        if inputs_dict['2048'].size(0) != 0:
            long_inputs = inputs_dict['2048'].to(config.device)
            optimizer.zero_grad()
            outputs = model(long_inputs, lengths[over_idx_2048]) # (B, 2)
            loss = criterion(outputs, labels[over_idx_2048])
            out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
            out_idx = torch.max(outputs, 1)[1]
            acc = torch.sum((out_idx == labels[over_idx_2048]) / out_idx.shape[0], dim=0).item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        if inputs_dict['4096'].size(0) != 0:
            long_inputs = inputs_dict['4096'].to(config.device)
            optimizer.zero_grad()
            outputs = model(long_inputs, lengths[over_idx_4096]) # (B, 2)
            loss = criterion(outputs, labels[over_idx_4096])
            out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
            out_idx = torch.max(outputs, 1)[1]
            acc = torch.sum((out_idx == labels[over_idx_4096]) / out_idx.shape[0], dim=0).item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            
        optimizer.zero_grad()

        ## 512 token idx 선별 후 0으로 만든 뒤 drop
        lengths[over_idx_1024] = 0
        lengths[over_idx_2048] = 0
        lengths[over_idx_4096] = 0
        lengths = lengths[lengths != 0]
        labels[over_idx_1024] = 999
        labels[over_idx_2048] = 999
        labels[over_idx_4096] = 999
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
    
    cum_loss, cum_acc, batch = 0, 0, 0 
    with torch.no_grad():
        for inputs_dict, labels, lengths, over_idx_1024, over_idx_2048, over_idx_4096 in dataloader:
            inputs, labels, lengths = inputs_dict['512'].to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
            
            if inputs_dict['1024'].size(0) != 0:
                long_inputs = inputs_dict['1024'].to(config.device)
                outputs = model(long_inputs, lengths[over_idx_1024]) # (B, 2)
                loss = criterion(outputs, labels[over_idx_1024])
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                acc = torch.sum((out_idx == labels[over_idx_1024]) / out_idx.shape[0], dim=0).item()
            if inputs_dict['2048'].size(0) != 0:
                long_inputs = inputs_dict['2048'].to(config.device)
                outputs = model(long_inputs, lengths[over_idx_2048]) # (B, 2)
                loss = criterion(outputs, labels[over_idx_2048])
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                acc = torch.sum((out_idx == labels[over_idx_2048]) / out_idx.shape[0], dim=0).item()
            if inputs_dict['4096'].size(0) != 0:
                long_inputs = inputs_dict['4096'].to(config.device)
                outputs = model(long_inputs, lengths[over_idx_4096]) # (B, 2)
                loss = criterion(outputs, labels[over_idx_4096])
                out_idx = torch.nn.functional.softmax(outputs.float(), dim=-1)
                out_idx = torch.max(outputs, 1)[1]
                acc = torch.sum((out_idx == labels[over_idx_4096]) / out_idx.shape[0], dim=0).item()

            lengths[over_idx_1024] = 0
            lengths[over_idx_2048] = 0
            lengths[over_idx_4096] = 0
            lengths = lengths[lengths != 0]
            labels[over_idx_1024] = 999
            labels[over_idx_2048] = 999
            labels[over_idx_4096] = 999
            labels = labels[labels != 999]
            outputs = model(inputs, lengths)
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
    parser.add_argument('-trim', '--d_trim', type=int, default=512, help='MHA trimmed dimension')
    parser.add_argument('-vocab', '--vocab_size', type=int, default=50265, help='Roberta-large Tokenizer Vocab sizes')
    
    # customizing hyperparms.
    parser.add_argument('--dataset', default='yelp', choices=['yelp', 'arxiv'], help='what kind of datasets are you using?')
    parser.add_argument('-pre', '--pretrained_weights', type=str, default='Roberta.pt', help='modified pretrained weights path')
    
    ## 실험모드 추후 반드시 수정할 것
    parser.add_argument('-B', '--batch_size', type=int, default=4, help='Batch sizes')
    parser.add_argument('-class', '--num_labels', type=int, default=7, help='# of classes in labels')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--device', type=str, default='cuda', help='If you want to use cuda, then write down cuda:0 or smt else')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate')
    config = parser.parse_args()
    
    ### dataset settings
    if config.dataset == 'yelp':
        paths = ['yelp_train.json', 'yelp_valid.json']
    else:
        paths = ['arxiv_train.json', 'arxiv_valid.json']
    
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

        valid_loss, valid_acc = evaluate(model, valid_dl, criterion, config)
        wandb.log({'valid_loss':valid_loss, 'valid_acc':valid_acc}) #TODO
        if valid_loss < best_valid_loss:
            save(os.path.join('checkpoint', f'model_{epoch+1:03d}.pt'), model, optimizer)
            best_valid_loss = valid_loss