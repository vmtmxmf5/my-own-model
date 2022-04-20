from transformers import RobertaTokenizer, RobertaModel
import torch
import collections


torch.set_printoptions(profile="full")


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')

weights = dict(model.state_dict().copy())
weights['embeddings.position_ids'] = torch.arange(0,4098).view(1, -1)
pos_emb = model.state_dict()['embeddings.position_embeddings.weight']

pad_emb = pos_emb[0:2, :]
rep_emb = pos_emb[2:,:].repeat(8,1) 
rev_emb = torch.cat((pad_emb, rep_emb), 0)

weights['embeddings.position_embeddings.weight'] = rev_emb


## name change
weights['encoder.embeddings.position_ids'] = weights.pop('embeddings.position_ids')
weights['encoder.embeddings.word_embeddings.weight'] = weights.pop('embeddings.word_embeddings.weight')
weights['encoder.embeddings.position_embeddings.weight'] = weights.pop('embeddings.position_embeddings.weight')
weights['encoder.embeddings.token_type_embeddings.weight'] = weights.pop('embeddings.token_type_embeddings.weight')
weights['encoder.embeddings.LayerNorm.weight'] = weights.pop('embeddings.LayerNorm.weight')
weights['encoder.embeddings.LayerNorm.bias'] = weights.pop('embeddings.LayerNorm.bias')

for i in range(24):
    if i == 0:
        weight_idx = torch.argsort(weights['encoder.embeddings.LayerNorm.weight'], descending=True)[:256]
        bias_idx = torch.argsort(weights['encoder.embeddings.LayerNorm.bias'], descending=True)[:256]
    if i >= 1:
        weight_idx = torch.argsort(weights[f'encoder.layer.{i-1}.intermediate.LayerNorm.weight'], descending=True)[:256]
        bias_idx = torch.argsort(weights[f'encoder.layer.{i-1}.intermediate.LayerNorm.bias'], descending=True)[:256]
    weights[f'encoder.layer.{i}.attention.Q_trim.weight'] = weights[f'encoder.layer.{i}.attention.self.query.weight'][weight_idx, :]
    # weights[f'encoder.layer.{i}.attention.Q_trim.bias'] = weights[f'encoder.layer.{i}.attention.self.query.bias'][bias_idx]
    weights[f'encoder.layer.{i}.attention.K_trim.weight'] = weights[f'encoder.layer.{i}.attention.self.key.weight'][weight_idx, :]
    # weights[f'encoder.layer.{i}.attention.K_trim.bias'] = weights[f'encoder.layer.{i}.attention.self.key.bias'][bias_idx]
    weights[f'encoder.layer.{i}.attention.V_trim.weight'] = weights[f'encoder.layer.{i}.attention.self.value.weight'][weight_idx, :]
    # weights[f'encoder.layer.{i}.attention.V_trim.bias'] = weights[f'encoder.layer.{i}.attention.self.value.bias'][bias_idx]
    

    weights[f'encoder.layer.{i}.attention.WQ.weight'] = weights.pop(f'encoder.layer.{i}.attention.self.query.weight')
    weights[f'encoder.layer.{i}.attention.WK.weight'] = weights.pop(f'encoder.layer.{i}.attention.self.key.weight')
    weights[f'encoder.layer.{i}.attention.WV.weight'] = weights.pop(f'encoder.layer.{i}.attention.self.value.weight')
    
    weights[f'encoder.layer.{i}.attention.WO.weight'] = weights.pop(f'encoder.layer.{i}.attention.output.dense.weight')
    weights[f'encoder.layer.{i}.attention.LayerNorm.weight'] = weights.pop(f'encoder.layer.{i}.attention.output.LayerNorm.weight')
    weights[f'encoder.layer.{i}.attention.LayerNorm.bias'] = weights.pop(f'encoder.layer.{i}.attention.output.LayerNorm.bias')

    weights[f'encoder.layer.{i}.intermediate.w_1.weight'] = weights.pop(f'encoder.layer.{i}.intermediate.dense.weight')
    weights[f'encoder.layer.{i}.intermediate.w_1.bias'] = weights.pop(f'encoder.layer.{i}.intermediate.dense.bias')
    weights[f'encoder.layer.{i}.intermediate.w_2.weight'] = weights.pop(f'encoder.layer.{i}.output.dense.weight')
    weights[f'encoder.layer.{i}.intermediate.w_2.bias'] = weights.pop(f'encoder.layer.{i}.output.dense.bias')
    weights[f'encoder.layer.{i}.intermediate.LayerNorm.weight'] = weights.pop(f'encoder.layer.{i}.output.LayerNorm.weight')
    weights[f'encoder.layer.{i}.intermediate.LayerNorm.bias'] = weights.pop(f'encoder.layer.{i}.output.LayerNorm.bias')

weights['encoder.dense.weight'] = weights.pop('pooler.dense.weight')
weights['encoder.dense.bias'] = weights.pop('pooler.dense.bias')

for i in range(24):
    weights.pop(f'encoder.layer.{i}.attention.self.query.bias')
    weights.pop(f'encoder.layer.{i}.attention.self.key.bias')
    weights.pop(f'encoder.layer.{i}.attention.self.value.bias')
    weights.pop(f'encoder.layer.{i}.attention.output.dense.bias')

plms = collections.OrderedDict(weights)


for m in plms:
    print(m, plms[f'{m}'].size())


path = 'Roberta.pt'
torch.save({'model': plms}, path)
