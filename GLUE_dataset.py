from transformers import LineByLineTextDataset, RobertaTokenizer
from datasets import load_dataset, get_dataset_config_names
import json
import torch

## sst ~ 63, qnli ~ 556, qqp ~ 275, imdb ~ 3099


configs = get_dataset_config_names('glue')
# ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

sst, qnli, qqp = 'sst2', 'qnli', 'qqp'


tr, val, test = [], [], []
max_len = 0
glue = load_dataset('glue', sst)
for a in glue['train']:
    a_tokens = tokenizer(a['sentence'])
    a_tokens['label'] = [a['label']]
    max_num = len(a_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    tr.append(dict(a_tokens))

for b in glue['validation']:
    b_tokens = tokenizer(b['sentence'])
    b_tokens['label'] = [b['label']]
    max_num = len(b_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    val.append(dict(b_tokens))

for c in glue['test']:
    c_tokens = tokenizer(c['sentence'])
    c_tokens['label'] = [c['label']]
    max_num = len(c_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    test.append(dict(c_tokens))
print(f'{sst} max token len: ', max_len)
with open(f'{sst}_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tr, ensure_ascii=False))
with open(f'{sst}_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(val, ensure_ascii=False))
with open(f'{sst}_test.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(test, ensure_ascii=False))


tr, val, test = [], [], []
max_len = 0
glue = load_dataset('glue', qnli)
for a in glue['train']:
    a_tokens = tokenizer(a['question'] + '</s>' + a['sentence'])
    # a_q = tokenizer(a['question'])
    a_tokens['label'] = [a['label']]
    # a_tokens['question'] = a_q['input_ids']
    max_num = len(a_tokens['input_ids'])

    if max_len < max_num:
        max_len = max_num

    if len(a_tokens['input_ids']) < 512:
        if len(a_tokens['question']) < 512:
            tr.append(dict(a_tokens))

for b in glue['validation']:
    b_tokens = tokenizer(b['question'] + '</s>' + b['sentence'])
    b_tokens['label'] = [b['label']]
    # b_tokens['question'] = b_q['input_ids']
    max_num = len(b_tokens['input_ids'])

    if max_len < max_num:
        max_len = max_num

    if len(b_tokens['input_ids']) < 512:
        if len(b_tokens['question']) < 512:
            val.append(dict(b_tokens))


for c in glue['test']:  
    c_tokens = tokenizer(c['question'] + '</s>' + c['sentence'])
    c_tokens['label'] = [c['label']]
    # c_tokens['question'] = c_q['input_ids']
    max_num = len(c_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num

    if len(c_tokens['input_ids']) < 512:
        if len(c_tokens['question']) < 512:
            test.append(dict(c_tokens))

print(f'{qnli} max token len: ', max_len)
with open(f'{qnli}_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tr, ensure_ascii=False))
with open(f'{qnli}_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(val, ensure_ascii=False))
with open(f'{qnli}_test.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(test, ensure_ascii=False))


tr, val, test = [], [], []
max_len = 0
glue = load_dataset('glue', qqp)
for a in glue['train']:
    a_tokens = tokenizer(a['question1'] + '</s>' + a['question2'])
    
    a_tokens['label'] = [a['label']]
    # a_tokens['question'] = a_q['input_ids']

    max_num = len(a_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    if len(a_tokens['input_ids']) < 512:
        if len(a_tokens['question']) < 512:
            tr.append(dict(a_tokens))

for b in glue['validation']:
    b_tokens = tokenizer(b['question1'] + '</s>' + b['question2'])
    # b_q = tokenizer(b['question2'])
    b_tokens['label'] = [b['label']]
    # b_tokens['question'] = b_q['input_ids']

    max_num = len(b_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    if len(b_tokens['input_ids']) < 512:
        if len(b_tokens['question']) < 512:
            val.append(dict(b_tokens))

for c in glue['test']:
    c_tokens = tokenizer(c['question1'] + '</s>' + c['question2'])
    
    c_tokens['label'] = [c['label']]
    # c_tokens['question'] = c_q['input_ids']

    max_num = len(c_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    if len(c_tokens['input_ids']) < 512:
        if len(c_tokens['question']) < 512:
            test.append(dict(c_tokens))
print(f'{qqp} max token len: ', max_len)
with open(f'{qqp}_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tr, ensure_ascii=False))
with open(f'{qqp}_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(val, ensure_ascii=False))
with open(f'{qqp}_test.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(test, ensure_ascii=False))


tr, val = [], []
imdb = load_dataset('imdb')
max_len = 0
for a in imdb['train']:
    a_tokens = tokenizer(a['text'])
    max_num = len(a_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num
    a_tokens['label'] = [a['label']]
    if len(a_tokens['input_ids']) < 512:
        tr.append(dict(a_tokens))

for b in imdb['test']:
    b_tokens = tokenizer(b['text'])
    max_num = len(b_tokens['input_ids'])
    if max_len < max_num:
        max_len = max_num

    b_tokens['label'] = [b['label']]

    if len(b_tokens['input_ids']) < 512:
        val.append(dict(b_tokens))
print('imdb max token len: ', max_len)

with open('imdb_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tr, ensure_ascii=False))

with open('imdb_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(val, ensure_ascii=False))
