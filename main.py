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
# import wandb #TODO


class KCCtrainds(Dataset):
    def __init__(self, path):
        super().__init__()
        with open(path, 'r', encoding='utf-8') as f:
            self.train = json.load(f)
    def __len__(self):
        return len(self.train)
    def __getitem__(self, index):
        line = self.train[index]
        label = line['label']
        return line['input_ids'], label

def collate_fn(batch):
    lines, labels = zip(*batch)
    max_len = len(max(lines))
    
    ids_res, label_res, lens = [], [], []
    for line, label in zip(lines, labels):
        tmp_len = len(line)
        len_ids = max_len - tmp_len
        ids_tensor = torch.cat([torch.LongTensor([line]), torch.LongTensor([[1] * len_ids])], dim=1)
        
        ids_res.append(ids_tensor)
        label_res.append(torch.LongTensor([label]))
        lens.append(tmp_len)
    ids_batch = torch.cat(ids_res, dim=0)
    label_batch = torch.cat(label_res, dim=0)
    len_batch = torch.LongTensor(lens)
    return ids_batch, label_batch, len_batch
     

def train(model, dataloader, criterion, optimizer, lr_scheduler, config, train_begin, epoch):
    model.train()
    begin = epoch_begin = time.time()
    cum_loss, cum_acc = 0, 0
    batch, print_batch = 0, 100
    total_num = len(dataloader)
    if config.dataset == 'imdb':
        for inputs, labels, lengths in dataloader:
            inputs, labels, lengths = inputs.to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
            optimizer.zero_grad()

            outputs = model(inputs, lengths) # (B, 2)
            loss = criterion(outputs, labels)
            acc = torch.sum((outputs == labels) / outputs.shape[0], dim=0).item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            cum_loss += loss.item()
            cum_acc += acc
    
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
            batch += 1
    return cum_loss / total_num


def evaluate(model, dataloader, criterion, config):
    model.eval() # Dropout 종료
    
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            inputs, labels, lengths = inputs.to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            acc = torch.sum((outputs == labels) / outputs.shape[0], dim=0).item()

if __name__=='__main__':
    # wandb.init(project='kcc', entity='vmtmxmf5') #TODO
    
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
    parser.add_argument('-B', '--batch_size', type=int, default=1, help='Batch sizes')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('-cuda', '--device', type=str, default='cpu', help='If you want to use cuda, then write down cuda:0 or smt else')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate')
    config = parser.parse_args()
    
    ### dataset settings
    if config.dataset == 'imdb':
        paths = ['imdb_train.json', 'imdb_valid.json']
    elif config.dataset == 'enwik8' or 'text8':
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
    train_ds = KCCtrainds(paths[0])
    
    train_dl = DataLoader(train_ds,
                        config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        collate_fn=collate_fn)

    model = ClassificationHead(config).to(config.device)
    pretrained_dict = torch.load(config.pretrained_weights, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict['model'])
    model.load_state_dict(model_dict)
    # wandb.watch(model, loss, log='all', log_freq=100) #TODO
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    num_training_steps = config.epochs * len(train_dl)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=8000,
                                num_training_steps=num_training_steps)

    for epoch in range(config.epochs):
        break
        # wandb.log({'train_loss':cum_loss / len(dataloader), 'train_acc':cum_acc}, step=100) #TODO
        