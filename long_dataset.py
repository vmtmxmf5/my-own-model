from datasets import load_dataset
from transformers import LongformerTokenizer
# import matplotlib.pyplot as plt
import json

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-large-4096')
yelp = load_dataset('yelp_review_full')

### 645 examples 밖에 없기 때문에 학습으로 부적절
# hyper = load_dataset('hyperpartisan_news_detection', 'byarticle')

# fig, axes = plt.subplots(1,2, figsize=(8,6))

yelp_train_len, yelp_valid_len, train, valid = [], [], [], []
for line in yelp['train']:
    tokens = tokenizer(line['text'])
    tokens['label'] = line['label']
    leng = len(tokens['input_ids'])
    yelp_train_len.append(leng)
    if leng < 4096:
        train.append(dict(tokens))
for line in yelp['test']:
    tokens = tokenizer(line['text'])
    tokens['label'] = line['label']
    leng = len(tokens['input_ids'])
    yelp_valid_len.append(leng)
    if leng < 4096:
        valid.append(dict(tokens))
with open('yelp_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(train, ensure_ascii=False))
with open('yelp_valid.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(valid, ensure_ascii=False))


# ax = axes.ravel()
# ax[0].hist(yelp_train_len)
# ax[0].set_title('Yelp-5 Train', fontsize=7)
# ax[0].set_xlabel('Roberta Tokenizer lengths')
# ax[1].hist(yelp_valid_len)
# ax[1].set_title('Yelp-5 Valdation', fontsize=7)
# ax[1].set_xlabel('Roberta Tokenizer lengths')
# plt.show()

# hyper_train_len, hyper_valid_len, train, valid = [], [], [], []
# for line in hyper['tr']



