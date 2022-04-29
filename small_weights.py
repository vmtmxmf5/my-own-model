from transformers import AutoTokenizer, AutoModel
import torch
import collections 


torch.set_printoptions(profile="full")

tokenzier = AutoTokenizer.from_pretrained('nreimers/BERT-Small-L-4_H-512_A-8')
model = AutoModel.from_pretrained('nreimers/BERT-Small-L-4_H-512_A-8')


# for m in model.state_dict():
#     print(m, model.state_dict()[f'{m}'].size())


weights = dict(model.state_dict().copy())

### 4096 
# weights['embeddings.position_ids'] = torch.arange(0,4098).view(1, -1)
# pos_emb = model.state_dict()['embeddings.position_embeddings.weight']
# pad_emb = pos_emb[0:2, :]
# rep_emb = pos_emb[2:,:].repeat(8,1) 
# rev_emb = torch.cat((pad_emb, rep_emb), 0)
# weights['embeddings.position_embeddings.weight'] = rev_emb


## name change
weights['encoder.embeddings.position_ids'] = weights.pop('embeddings.position_ids')
weights['encoder.embeddings.word_embeddings.weight'] = weights.pop('embeddings.word_embeddings.weight')
weights['encoder.embeddings.position_embeddings.weight'] = weights.pop('embeddings.position_embeddings.weight')
weights['encoder.embeddings.token_type_embeddings.weight'] = weights.pop('embeddings.token_type_embeddings.weight')
weights['encoder.embeddings.LayerNorm.weight'] = weights.pop('embeddings.LayerNorm.weight')
weights['encoder.embeddings.LayerNorm.bias'] = weights.pop('embeddings.LayerNorm.bias')


for i in range(4):
    # if i == 0:
    #     weight_idx = torch.argsort(weights['encoder.embeddings.LayerNorm.weight'], descending=True)
    #     bias_idx = torch.argsort(weights['encoder.embeddings.LayerNorm.bias'], descending=True)
    # if i >= 1:
    #     weight_idx = torch.argsort(weights[f'encoder.layer.{i-1}.intermediate.LayerNorm.weight'], descending=True)
    #     bias_idx = torch.argsort(weights[f'encoder.layer.{i-1}.intermediate.LayerNorm.bias'], descending=True)
    
    weights[f'encoder.layer.{i}.attention.Q.weight_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.query.weight')
    weights[f'encoder.layer.{i}.attention.Q.weight_s2'] = weights[f'encoder.layer.{i}.attention.Q.weight_s1']
    weights[f'encoder.layer.{i}.attention.Q.bias_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.query.bias')
    weights[f'encoder.layer.{i}.attention.Q.bias_s2'] = weights[f'encoder.layer.{i}.attention.Q.bias_s1']
    
    weights[f'encoder.layer.{i}.attention.K.weight_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.key.weight')
    weights[f'encoder.layer.{i}.attention.K.weight_s2'] = weights[f'encoder.layer.{i}.attention.K.weight_s1']
    weights[f'encoder.layer.{i}.attention.K.bias_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.key.bias')
    weights[f'encoder.layer.{i}.attention.K.bias_s2'] = weights[f'encoder.layer.{i}.attention.K.bias_s1']
    
    weights[f'encoder.layer.{i}.attention.V.weight_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.value.weight')
    weights[f'encoder.layer.{i}.attention.V.weight_s2'] = weights[f'encoder.layer.{i}.attention.V.weight_s1']
    weights[f'encoder.layer.{i}.attention.V.bias_s1'] = weights.pop(f'encoder.layer.{i}.attention.self.value.bias')
    weights[f'encoder.layer.{i}.attention.V.bias_s2'] = weights[f'encoder.layer.{i}.attention.V.bias_s1']

    weights[f'encoder.layer.{i}.attention.O.weight_s1'] = weights.pop(f'encoder.layer.{i}.attention.output.dense.weight')
    weights[f'encoder.layer.{i}.attention.O.weight_s2'] = weights[f'encoder.layer.{i}.attention.O.weight_s1']
    weights[f'encoder.layer.{i}.attention.O.bias_s1'] = weights.pop(f'encoder.layer.{i}.attention.output.dense.bias')
    weights[f'encoder.layer.{i}.attention.O.bias_s2'] = weights[f'encoder.layer.{i}.attention.O.bias_s1']

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

# for i in range(4):
#     weights.pop(f'encoder.layer.{i}.attention.self.query.bias')
#     weights.pop(f'encoder.layer.{i}.attention.self.key.bias')
#     weights.pop(f'encoder.layer.{i}.attention.self.value.bias')
#     weights.pop(f'encoder.layer.{i}.attention.output.dense.bias')

plms = collections.OrderedDict(weights)


for m in plms:
    print(m, plms[f'{m}'].size())


path = 'Small.pt'
torch.save({'model': plms}, path)
