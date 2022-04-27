import json
import random
from transformers import LongformerTokenizer
# import matplotlib.pyplot as plt

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
# fig, axes = plt.subplots(1,1, figsize=(8,6))

train, valid = [], []
arxiv_len = []
with open('arxiv-metadata-oai-snapshot.json') as f:
    for line in f:
        line_data = {'text':[], 'label':[]}
        tmp = json.loads(line)
        line_data['text'] = tmp['abstract']
        tokens = tokenizer(line_data['text'])
        tokens['label'] = tmp['categories']
        leng = len(tokens['input_ids'])
        arxiv_len.append(leng)
        if leng < 4096:
            if random.random() < 0.85:
                train.append(dict(tokens))
            else:
                valid.append(dict(tokens))

with open('arxiv_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(train, ensure_ascii=False))

with open('arxiv_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid, ensure_ascii=False))

# ax = axes.ravel()
# ax[0].hist(arxiv_len)
# ax[0].set_title('Arxiv', fontsize=7)
# ax[0].set_xlabel('Roberta Tokenizer lengths')
# plt.show()