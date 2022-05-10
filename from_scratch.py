from transformers import EncoderDecoderModel

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from transformers import get_scheduler
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

import json
import argparse
import time
import os
import wandb #TODO


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class ActivationFunction(object):
    relu = "relu"
    gelu = "gelu"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.relu: F.relu,
    ActivationFunction.gelu: F.gelu,
}


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        activation_fn (ActivationFunction): activation function used.
    """

    def __init__(self, d_model, d_ff, dropout=0.1,
                 activation_fn=ActivationFunction.relu):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.activation = ACTIVATION_FUNCTIONS[activation_fn]
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.activation(self.w_1(self.LayerNorm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions
        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        if self.max_relative_positions > 0 and attn_type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(key.device))

        query = shape(query)
        
        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))
        
        if self.max_relative_positions > 0 and attn_type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and attn_type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)

        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    1) mask : 정상적인 input은 1, padding은 0으로 변환
    2) incremental_indices : padding 전까지만 torch.arange
    3) return : padding idx 이후부터 position_ids 세도록 전환 + padding은 1
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx        
        
    
class embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.token_type_embeddings = nn.Embedding(1, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.padding_idx = config.pad_token_id

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        ## input_ids : (B, T)
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :class:`onmt.Models.NMTModel`.
    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        raise NotImplementedError

    def _check_args(self, src, lengths=None, hidden=None):
        n_batch = src.size(1)
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        """
        Args:
            src (LongTensor):
               padded sequences of sparse indices ``(src_len, batch, nfeat)``
            lengths (LongTensor): length of each sequence ``(batch,)``
        Returns:
            (FloatTensor, FloatTensor, FloatTensor):
            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
            * lengths
        """

        raise NotImplementedError


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(heads, d_model, dropout, max_relative_positions=0) ### shared weights 쓰고 싶으면 MHA로 변경할것
        self.intermediate = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
        Returns:
            (FloatTensor):
            * outputs ``(batch_size, src_len, model_dim)``
        """
        # input_norm = self.LayerNorm(inputs)
        context = self.attention(inputs, inputs, inputs,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.intermediate(out)

    def update_dropout(self, dropout, attention_dropout):
        self.attention.update_dropout(attention_dropout)
        self.intermediate.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embeddings = embeddings(config)
        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(
                config.d_model, config.heads, config.d_ff, config.dropout)
             for i in range(config.num_layers)])
        # self.LayerNorm = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dense = nn.Linear(config.d_model, config.d_model)
    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            # opt.max_relative_positions,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
        )

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        # self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for lay in self.layer:
            out = lay(out, mask)
        # out = self.LayerNorm(out)
        out = self.dense(out)
        return emb, out, lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)



class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, inputs, lengths):
        outputs = self.encoder(inputs, lengths)
        _, x, __ = outputs
        x = x[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.out_proj(x)
        return x            


class Config:
    def __init__(self,
                 num_layers : int  = 12,
                 d_model : int = 768,
                 heads : int = 12,
                 d_ff : int = 2048,
                 dropout : float = 0.1,
                 attention_dropout : float = 0.1,
                 pad_token_id : int = 1,
                 max_position_embeddings : int = 4096 + 2,
                 num_labels : int = 2,
                 vocab_size = 50265
                ):
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.num_labels = num_labels

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
    max_len = max(lengths)

    ids_res_256 = []
    for i, (line, lens) in enumerate(zip(lines, lengths)):
        len_ids = max_len - lens
        padding = torch.ones((1, abs(len_ids)), dtype=torch.long)
        ids_tensor = torch.cat([torch.LongTensor([line]), padding], dim=1)
        ids_res_256.append(ids_tensor)
 
    ids_batch_256 = torch.cat(ids_res_256, dim=0)
    label_batch = torch.LongTensor(labels).reshape(-1)
    len_batch = torch.LongTensor(lengths)
    return ids_batch_256, label_batch, len_batch
     

def train(model, dataloader, criterion, optimizer, lr_scheduler, config, train_begin, epoch):
    model.train()
    begin = epoch_begin = time.time()
    cum_loss, cum_acc = 0, 0
    batch, print_batch = 0, 100
    total_num = len(dataloader)

    for inputs_dict, labels, lengths in dataloader:
        if inputs_dict.shape[1] < 512:
            inputs, labels, lengths = inputs_dict.to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
                
            optimizer.zero_grad()

            ## 512 token idx 선별 후 0으로 만든 뒤 drop
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
        for inputs_dict, labels, lengths, over_idx in dataloader:
            if inputs_dict.shape[1] < 512:
                inputs, labels, lengths = inputs_dict.to(config.device), labels.to(config.device), lengths.to(config.device) # multi-gpu 사용시 제거
                
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
    parser.add_argument('--dataset', default='imdb', choices=['sst2', 'qqp', 'qnli', 'enwik8', 'text8', 'imdb'], help='what kind of datasets are you using?')

    parser.add_argument('-B', '--batch_size', type=int, default=16, help='Batch sizes')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('-cuda', '--device', type=str, default='cuda', help='If you want to use cuda, then write down cuda:0 or smt else')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate')
    config = parser.parse_args()
    basic_config = Config()

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


    model = ClassificationHead(basic_config).to(config.device)
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
