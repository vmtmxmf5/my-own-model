from os import lseek
import torch
import torch.nn as nn

from small_module import AutoMHA
from small_module import PositionwiseFeedForward
from small_module import ActivationFunction
from small_module import sequence_mask
from small_module import embeddings
from transformers import LongformerModel, AutoModel, AutoTokenizer


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

    def __init__(self, d_model, heads, d_ff, dropout, d_trim,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = AutoMHA(heads, d_model, d_trim, dropout)
        self.intermediate = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)
        # self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

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
                config.d_model, config.heads, config.d_ff, config.dropout, config.d_trim)
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


class LongformerHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.longformer = RobertaModel.from_pretrained('roberta-large')
#         self.longformer = LongformerModel.from_pretrained('allenai/longformer-large-4096')
#         self.tokenzier = RobertaTokenizer.from_pretrained('roberta-large')
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, inputs):
        # inputs = self.tokenzier(inputs)
        # inputs = {'input_ids':inputs}
        outputs = self.longformer(**inputs)
        x = outputs.pooler_output
        x = self.out_proj(x)
        return x        

    

class Config:
    def __init__(self,
                 num_layers : int  = 24,
                 d_model : int = 1024,
                 heads : int = 16,
                 d_ff : int = 4096,
                 dropout : float = 0.1,
                 attention_dropout : float = 0.1,
                 pad_token_id : int = 1,
                 max_position_embeddings : int = 4096 + 2,
                 num_labels : int = 2,
                 d_trim : int = 256,
                 vocab_size = None
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
        self.d_trim = d_trim
        
if __name__=='__main__':
    config = Config(vocab_size=30)
    model = ClassificationHead(config)
    # model = LongformerHead(config)

    # for m in model.state_dict():
    #   print(m, model.state_dict()[f'{m}'].size())

    dummy_inputs = torch.randint(2, 30, (3,5))
    dummy_inputs[1:, -2:] =  1
    lengths = torch.LongTensor((5, 3, 3))
    pred = model(dummy_inputs, lengths)
    # pred = model(dummy_inputs)
    print(pred)