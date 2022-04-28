from ast import arg
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


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


# class LinearFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, key_len, stage, bias=None):
#         # weight = weight.t()
#         if key_len <= 512:
#             final_weight = weight[:, :1 * stage]
#         if 512 < key_len <= 1024:
#             final_weight = weight[:, :2 * stage]
#         if 1024 < key_len <= 2048:
#             final_weight = weight[:, :3 * stage]
#         if 2048 < key_len < 4096:
#             final_weight = weight[:, :4 * stage]
#         tmp = torch.LongTensor([key_len, stage])
#         ctx.save_for_backward(input, weight, final_weight.t(), bias, tmp) # TODO
#         # print(input.shape)
#         output = input.matmul(final_weight) ### TODO
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, final_weight, bias, tmp = ctx.saved_tensors #TODO
#         grad_input = grad_weight = grad_bias = None
#         # print('backprop w.shape : ', weight.shape)
#         # print('grad out : ', grad_output.shape)
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.matmul(final_weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.permute(0,2,1).matmul(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)
#         # print(final_weight.size()) # TODO
#         grad_weight = torch.sum(grad_weight, 0) # TODO
#         weight = torch.zeros(weight.size()).to(input.device)
#         if int(tmp[0]) <= 512:
#             weight[:1 * int(tmp[1]), :] = grad_weight
#         if 512 < int(tmp[0]) <= 1024:
#             weight[1 * int(tmp[1]):2 * int(tmp[1]), :] = grad_weight[1 * int(tmp[1]):2 * int(tmp[1]), :]
#         if 1024 < int(tmp[0]) <= 2048:
#             weight[2 * int(tmp[1]):3 * int(tmp[1]), :] = grad_weight[2 * int(tmp[1]):3 * int(tmp[1]), :]
#         if 2048 < int(tmp[0]) < 4096:
#             weight[3 * int(tmp[1]):4 * int(tmp[1]), :] = grad_weight[3 * int(tmp[1]):4 * int(tmp[1]), :]
        
#         return grad_input, weight.t(), grad_bias, None # TODO


# class FinalFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, key_len, stage, bias=None):
#         # weight = weight.t() ### TODO
#         # print(weight.shape)
#         if key_len <= 512:
#             final_weight = weight[:1 * stage, :] ### TODO
#         if 512 < key_len <= 1024:
#             final_weight = weight[:2 * stage, :] ### TODO
#         if 1024 < key_len <= 2048:
#             final_weight = weight[:3 * stage, :] ### TODO
#         if 2048 < key_len < 4096:
#             final_weight = weight[:4 * stage, :] ### TODO
#         # final weight to devcie??
#         tmp = torch.LongTensor([key_len, stage])
#         ctx.save_for_backward(input, weight, final_weight.t(), bias, tmp) 

#         output = input.matmul(final_weight) ### TODO
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, org_weight, final_weight, bias, tmp = ctx.saved_tensors 
#         grad_input = grad_weight = grad_bias = None
#         # print('backprop w.shape : ', weight.shape)
#         # print(grad_output.shape)
#         # print(weight.shape)
        
#         weight = torch.zeros(org_weight.t().size()).to(input.device) ## TODO
#         # print(weight.shape)
#         # print(final_weight.shape)
#         if int(tmp[0]) <= 512:
#             weight[:, :1 * int(tmp[1])] = final_weight ### TODO
#             weights = weight[:, :1 * int(tmp[1])]
#         if 512 < int(tmp[0]) <= 1024:
#             weight[:, 1 * int(tmp[1]):2 * int(tmp[1])] = final_weight[:, 1 * int(tmp[1]):] ### TODO
#             weights = weight[:, :2 * int(tmp[1])]
#         if 1024 < int(tmp[0]) <= 2048:
#             weight[:, 2 * int(tmp[1]):3 * int(tmp[1])] = final_weight[:, 2 * int(tmp[1]):] ### TODO
#             weights = weight[:, :3 * int(tmp[1])]
#         if 2048 < int(tmp[0]) <= 4096:
#             weight[:, 3 * int(tmp[1]):4 * int(tmp[1])] = final_weight[:, 3 * int(tmp[1]):] ### TODO
#             weights = weight[:, :4 * int(tmp[1])]
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.matmul(weights) ## TODO
#             # print(grad_input.shape, grad_output.shape)
#         if ctx.needs_input_grad[1]:
#             # grad_weight = grad_output.t().matmul(input)
#             grad_weight = grad_output.permute(0, 2, 1).matmul(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)
#         grad_weight = torch.sum(grad_weight, 0).t() # TODO 
#         # print(grad_weight)
        
#         grad_weight_size = grad_weight.size(0)
#         # print(weight.shape, grad_weight.size())
#         weight = weight.t()
#         if int(tmp[0]) <= 512:
#             weight[:1 * int(tmp[1]), :] = grad_weight ### TODO
#         if 512 < int(tmp[0]) <= 1024:
#             weight[1 * int(tmp[1]):2 * int(tmp[1]), :] = grad_weight[1 * int(tmp[1]):2 * int(tmp[1]), :]
#         if 1024 < int(tmp[0]) <= 2048:
#             weight[2 * int(tmp[1]):3 * int(tmp[1]), :] = grad_weight[2 * int(tmp[1]):3 * int(tmp[1]), :]
#         if 2048 < int(tmp[0]) < 4096:
#             weight[3 * int(tmp[1]):4 * int(tmp[1]), :] = grad_weight[3 * int(tmp[1]):4 * int(tmp[1]), :]
#         return grad_input, weight, grad_bias, None ## TODO


### 512 token 미만용
# class LinearFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, key_len, stage):
#         if key_len <= 256:
#             final_weight = weight[:, :1 * stage]
#         else:
#             final_weight = weight[:, 1 * stage:]
#         tmp = torch.LongTensor([key_len, stage])
#         ctx.save_for_backward(input, final_weight, bias, tmp) # TODO
#         # print(input.shape)
#         output = input.matmul(final_weight.t()) ### TODO
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, final_weight, bias, tmp = ctx.saved_tensors #TODO
#         grad_input = grad_weight = grad_bias = None
#         # print('backprop w.shape : ', weight.shape)
#         # print('grad out : ', grad_output.shape)
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.matmul(final_weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.permute(0,2,1).matmul(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)
#         # print(final_weight.size()) # TODO
#         grad_weight = torch.sum(grad_weight, 0) # TODO
        
#         d_r, d_c = grad_weight.size()
#         weight = torch.zeros(d_r, d_c * 2).to(input.device)
#         if int(tmp[0]) <= 256:
#             weight[:, :1 * int(tmp[1])] = grad_weight
#         else:
#             weight[:, 1 * int(tmp[1]):2 * int(tmp[1])] = grad_weight
        
#         return grad_input, weight, grad_bias, None, None # TODO


# class FinalFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, key_len, stage):
#         # weight = weight.t() ### TODO
#         # print(weight.shape)
#         if key_len <= 256:
#             final_weight = weight[:1 * stage, :] ### TODO
#         else:
#             final_weight = weight[1 * stage:, :] ### TODO
#         # final weight to devcie??
#         tmp = torch.LongTensor([key_len, stage])
#         ctx.save_for_backward(input, final_weight, bias, tmp) 

#         output = input.matmul(final_weight.t()) ### TODO
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, final_weight, bias, tmp = ctx.saved_tensors 
#         grad_input = grad_weight = grad_bias = None
#         # print('backprop w.shape : ', weight.shape)
#         # print(grad_output.shape)
#         # print(weight.shape)
        
#         # weight = torch.zeros(org_weight.t().size()).to(input.device) ## TODO
#         # print(weight.shape)
#         # print(final_weight.shape)
#         # if int(tmp[0]) <= 256:
#         #     weight[:, :1 * int(tmp[1])] = final_weight ### TODO
#         #     weights = weight[:, :1 * int(tmp[1])]
#         # else:
#         #     weight[:, 1 * int(tmp[1]):2 * int(tmp[1])] = final_weight[:, :1 * int(tmp[1])] ### TODO
#         #     weights = weight[:, :2 * int(tmp[1])]
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.matmul(final_weight) ## TODO
#             # print(grad_input.shape, grad_output.shape)
#         if ctx.needs_input_grad[1]:
#             # grad_weight = grad_output.t().matmul(input)
#             grad_weight = grad_output.permute(0, 2, 1).matmul(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)
#         grad_weight = torch.sum(grad_weight, 0) # TODO 
#         # print(grad_weight)
        
#         d_r, d_c = grad_weight.size()
#         weight = torch.zeros(d_r * 2, d_c).to(input.device)
#         if int(tmp[0]) <= 256:
#             weight[:1 * int(tmp[1]), :] = grad_weight
#         else:
#             weight[1 * int(tmp[1]):2 * int(tmp[1]), :] = grad_weight
#         return grad_input, weight, grad_bias, None, None ## TODO

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.permute(0,2,1).matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class FinalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.permute(0,2,1).matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class AutoMHA(nn.Module):
    def __init__(self,
                nhead,
                d_model,
                d_trim,
                dropout=0.1,
                learnable_mode=True):
        assert d_model % nhead == 0
        super().__init__()
        self.head_dim = d_model // nhead ### TODO
        self.d_model = d_model

        self.nhead = nhead

        # nn.parameter를 사용하면 weight을 backward에 쓰겠다는 의미
        self.WQ_256 = nn.Parameter(torch.randn(d_model, d_model))
        self.bQ_256 = nn.Parameter(torch.randn(d_model))
        self.WK_256 = nn.Parameter(torch.randn(d_model, d_model))
        self.bK_256 = nn.Parameter(torch.randn(d_model))
        self.WV_256 = nn.Parameter(torch.randn(d_model, d_model))
        self.bV_256 = nn.Parameter(torch.randn(d_model))

        self.WQ_512 = nn.Parameter(torch.randn(d_model, d_model))
        self.bQ_512 = nn.Parameter(torch.randn(d_model))
        self.WK_512 = nn.Parameter(torch.randn(d_model, d_model))
        self.bK_512 = nn.Parameter(torch.randn(d_model))
        self.WV_512 = nn.Parameter(torch.randn(d_model, d_model))
        self.bV_512 = nn.Parameter(torch.randn(d_model))
        self.K = LinearFunction.apply
        self.V = LinearFunction.apply
        self.Q = LinearFunction.apply

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.WO_256 = nn.Parameter(torch.randn(d_model, d_model)) ### TODO
        self.bO_256 = nn.Parameter(torch.randn(d_model))
        self.WO_512 = nn.Parameter(torch.randn(d_model, d_model)) ### TODO
        self.bO_512 = nn.Parameter(torch.randn(d_model))
        self.O = FinalFunction.apply

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None, attn_type=None):
        # query : (B, query_len, d_model) // FloatTensor
        # key   : (B, key_len, d_model) // FloatTensor
        # value : (B, seq_len, d_model) // FloatTensor
        # mask  : (B, 1, src_len) == (B, query_len, key_len)
        #       : (B, tgt_len, src_len) == (B, query_len, key_len)

        B = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            '''Projection'''
            # output : (B, nhead, seq_len, head_dim)
            return x.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        def unshape(x):
            '''Compute context'''
            # output : (B, seq_len, d_model)
            return x.transpose(1, 2).contiguous().view(B, -1, self.nhead * self.head_dim)


        def replace_outputs(final_weight, final_bias, start=0, stage=None):
            with torch.no_grad():
                final_weight = self.shared_final_weight[:(start + 1) * stage, :]
                final_bias = self.shared_final_bias[:(start + 1) * stage]

        # stage = int(self.d_model
        if key_len < 256:
            Q = self.Q(query, self.WQ_256, key_len, self.d_model, self.bQ_256)
            K = self.K(key, self.WK_256, key_len, self.d_model, self.bK_256)
            V = self.V(value, self.WV_256, key_len, self.d_model, self.bV_256)
        else:
            Q = self.Q(query, self.WQ_512, key_len, self.d_model, self.bQ_512)
            K = self.K(key, self.WK_512, key_len, self.d_model, self.bK_512)
            V = self.V(value, self.WV_512, key_len, self.d_model, self.bV_512)
        
 
        Q, K, V = shape(Q), shape(K), shape(V)
        
        ## 2) Calculate scores
        Q = Q / math.sqrt(self.head_dim)
        # QK : (B, nhead, query_len, key_len)
        QK = torch.matmul(Q, K.transpose(2, 3))
        scores = QK.float()
        
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, 1 or query_len, key_len)
            scores = scores.masked_fill(mask, -1e18)
        
        ## 3) Apply dropout and compute context vector
        attn = self.softmax(scores).to(query.dtype) # (B, nhead, query_len, key_len)
        drop_attn = self.dropout(attn)

        # (B, nhead, query_len, key_len) @ (B, nhead, value_len, head_dim)

        context_original = torch.matmul(drop_attn, V)
        # (B, nhead, query_len, head_dim)

        context = unshape(context_original) # (B, q_len, d_model)
        if key_len < 256:
            output = self.O(context, self.WO_256, key_len, self.d_model, self.bO_256) # (B, q_len, d_model)
        else:
            output = self.O(context, self.WO_512, key_len, self.d_model, self.bO_512) # (B, q_len, d_model)
        # print(output.shape)
        output = self.LayerNorm(output)
        
        # attns = attn.view(B, self.nhead, query_len, key_len)
        return output #, attns
    
    def update_dropout(self, dropout):
        self.dropout.p = dropout
    
    # backward 계산할 때 gradient 바꿔치기
    # backward 계산 후 update 이전에 바꿔치기


class MHA(nn.Module):
    def __init__(self,
                nhead,
                d_model,
                d_trim,
                dropout=0.1,
                learnable_mode=True):
        assert d_model % nhead == 0
        super().__init__()
        self.head_dim = d_model // nhead
        self.d_model = d_model

        self.nhead = nhead
        self.d_trim = d_trim
        
        if learnable_mode: # (gradient가 업데이트 되는 과정을 델타 * R 로 표현한 것 같은데)
            self.Q_trim = nn.Linear(d_model, d_trim)
            self.K_trim = nn.Linear(d_model, d_trim)
            self.V_trim = nn.Linear(d_model, d_trim)
        else:
            self.Q_trim = torch.randn(d_trim, d_model, requires_grad=False)
            self.K_trim = torch.randn(d_trim, d_model, requires_grad=False)
            self.V_trim = torch.randn(d_trim, d_model, requires_grad=False)

        # nn.parameter를 사용하면 weight을 backward에 쓰겠다는 의미
        self.shared_weights_Q = torch.randn(nhead * self.head_dim, d_trim)
        self.shared_bias_Q = torch.randn(nhead * self.head_dim)
        self.shared_weights_K = torch.randn(nhead * self.head_dim, d_trim)
        self.shared_bias_K = torch.randn(nhead * self.head_dim)
        self.shared_weights_V = torch.randn(nhead * self.head_dim, d_trim)
        self.shared_bias_V = torch.randn(nhead * self.head_dim)

        self.Q1 = nn.Linear(d_trim, (nhead // 4) * self.head_dim)
        self.K1 = nn.Linear(d_trim, (nhead // 4) * self.head_dim)
        self.V1 = nn.Linear(d_trim, (nhead // 4) * self.head_dim)

        self.Q2 = nn.Linear(d_trim, 2 * (nhead // 4) * self.head_dim)
        self.K2 = nn.Linear(d_trim, 2 * (nhead // 4) * self.head_dim)
        self.V2 = nn.Linear(d_trim, 2 * (nhead // 4) * self.head_dim)

        self.Q3 = nn.Linear(d_trim, 3 * (nhead // 4) * self.head_dim)
        self.K3 = nn.Linear(d_trim, 3 * (nhead // 4) * self.head_dim)
        self.V3 = nn.Linear(d_trim, 3 * (nhead // 4) * self.head_dim)

        self.Q4 = nn.Linear(d_trim, 4 * (nhead // 4) * self.head_dim)
        self.K4 = nn.Linear(d_trim, 4 * (nhead // 4) * self.head_dim)
        self.V4 = nn.Linear(d_trim, 4 * (nhead // 4) * self.head_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.final_linear_1 = nn.Linear((nhead // 4) * self.head_dim, d_model)
        self.final_linear_2 = nn.Linear(2 * (nhead // 4) * self.head_dim, d_model)
        self.final_linear_3 = nn.Linear(3 * (nhead // 4) * self.head_dim, d_model)
        self.final_linear_4 = nn.Linear(4 * (nhead // 4) * self.head_dim, d_model)
        self.shared_final_weight = torch.randn(d_model, nhead * self.head_dim)
        self.shared_final_bias = torch.randn(nhead * self.head_dim)

        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None, attn_type=None):
        # query : (B, query_len, d_model) // FloatTensor
        # key   : (B, key_len, d_model) // FloatTensor
        # value : (B, seq_len, d_model) // FloatTensor
        # mask  : (B, 1, src_len) == (B, query_len, key_len)
        #       : (B, tgt_len, src_len) == (B, query_len, key_len)

        B = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x, nhead):
            '''Projection'''
            # output : (B, nhead, seq_len, head_dim)
            return x.view(B, -1, nhead, self.head_dim).transpose(1, 2)
        
        def unshape(x, nhead):
            '''Compute context'''
            # output : (B, seq_len, d_model)
            return x.transpose(1, 2).contiguous().view(B, -1, nhead * self.head_dim)
        
        def replace_params(Q_weight, Q_bias, K_weight, K_bias, V_weight, V_bias,
                        start = 0, stage = None):
            # 2-stage : start 0
            # 3-stage : start 1
            # 4-stage : start 2
            with torch.no_grad():
                Q_weight = self.shared_weights_Q[:(start + 1) * stage, :]
                Q_bias = self.shared_bias_Q[:(start + 1) * stage]
                K_weight = self.shared_weights_K[:(start + 1) * stage, :]
                K_bias = self.shared_bias_K[:(start + 1) * stage]
                V_weight = self.shared_weights_V[:(start + 1) * stage, :]
                V_bias = self.shared_bias_V[:(start + 1) * stage]
        
        def save_params(stage):
            with torch.no_grad():
                self.shared_weights_Q[:stage, :] = self.Q1.weight[:stage, :]
                self.shared_bias_Q[:stage] = self.Q1.bias[:stage]
                self.shared_weights_K[:stage, :] = self.K1.weight[:stage, :]
                self.shared_bias_K[:stage] = self.K1.bias[:stage]
                self.shared_weights_V[:stage, :] = self.V1.weight[:stage, :]
                self.shared_bias_V[:stage] = self.V1.bias[:stage]
                
                self.shared_weights_Q[stage:2*stage, :] = self.Q2.weight[stage:2*stage, :]
                self.shared_bias_Q[stage:2*stage] = self.Q2.bias[stage:2*stage]
                self.shared_weights_K[stage:2*stage, :] = self.K2.weight[stage:2*stage, :]
                self.shared_bias_K[stage:2*stage] = self.K2.bias[stage:2*stage]
                self.shared_weights_V[stage:2*stage, :] = self.V2.weight[stage:2*stage, :]
                self.shared_bias_V[stage:2*stage] = self.V2.bias[stage:2*stage]
                
                self.shared_weights_Q[2*stage:3*stage, :] = self.Q3.weight[2*stage:3*stage, :]
                self.shared_bias_Q[2*stage:3*stage] = self.Q3.bias[2*stage:3*stage]
                self.shared_weights_K[2*stage:3*stage, :] = self.K3.weight[2*stage:3*stage, :]
                self.shared_bias_K[2*stage:3*stage] = self.K3.bias[2*stage:3*stage]
                self.shared_weights_V[2*stage:3*stage, :] = self.V3.weight[2*stage:3*stage, :]
                self.shared_bias_V[2*stage:3*stage] = self.V3.bias[2*stage:3*stage]
                
                self.shared_weights_Q[3*stage:, :] = self.Q4.weight[3*stage:, :]
                self.shared_bias_Q[3*stage:] = self.Q4.bias[3*stage:]
                self.shared_weights_K[3*stage:, :] = self.K4.weight[3*stage:, :]
                self.shared_bias_K[3*stage:] = self.K4.bias[3*stage:]
                self.shared_weights_V[3*stage:, :] = self.V4.weight[3*stage:, :]
                self.shared_bias_V[3*stage:] = self.V4.bias[3*stage:]

                self.shared_final_weight[:, :stage] = self.final_linear_1.weight[:, :stage]
                self.shared_final_bias[:stage] = self.final_linear_1.bias[:stage]
                self.shared_final_weight[:, stage:2*stage] = self.final_linear_2.weight[:, stage:2*stage]
                self.shared_final_bias[stage:2*stage] = self.final_linear_2.bias[stage:2*stage]
                self.shared_final_weight[:, stage:3*stage] = self.final_linear_3.weight[:, stage:3*stage]
                self.shared_final_bias[stage:3*stage] = self.final_linear_3.bias[stage:3*stage]
                self.shared_final_weight[:, stage:4*stage] = self.final_linear_4.weight[:, stage:4*stage]
                self.shared_final_bias[stage:4*stage] = self.final_linear_4.bias[stage:4*stage]


        def replace_outputs(final_weight, final_bias, start=0, stage=None):
            with torch.no_grad():
                final_weight = self.shared_final_weight[:(start + 1) * stage, :]
                final_bias = self.shared_final_bias[:(start + 1) * stage]

        stage = (self.nhead // 4) * self.head_dim
        save_params(stage)
        
        query = self.Q_trim(query)
        key = self.K_trim(key)
        value = self.V_trim(value)

        ## 1) Project Q, K, V
        if key_len <= 256:
            Q = self.Q1(query)
            K = self.K1(key)
            V = self.V1(value)
            
            # Q : (B, nhead, query_len, head_dim)
            nhead = self.nhead // 4
            Q, K, V = shape(Q, nhead), shape(K, nhead), shape(V, nhead)
            
            mode = 'mode 1'

        elif 256 < key_len <= 512:
            replace_params(self.Q2.weight,
                        self.Q2.bias,
                        self.K2.weight,
                        self.K2.bias,
                        self.V2.weight,
                        self.V2.bias,
                        1,stage)  
            Q = self.Q2(query)
            K = self.K2(key)
            V = self.V2(value)

            nhead = (self.nhead // 4) * 2
            Q, K, V = shape(Q, nhead), shape(K, nhead), shape(V, nhead)

            mode = 'mode 2'

        elif 512 < key_len <= 2048:
            replace_params(self.Q3.weight,
                        self.Q3.bias,
                        self.K3.weight,
                        self.K3.bias,
                        self.V3.weight,
                        self.V3.bias,
                        2,stage)    
            Q = self.Q3(query)
            K = self.K3(key)
            V = self.V3(value)

            nhead = (self.nhead // 4) * 3
            Q, K, V = shape(Q, nhead), shape(K, nhead), shape(V, nhead)  
            
            mode = 'mode 3'

        else:
            replace_params(self.Q4.weight,
                        self.Q4.bias,
                        self.K4.weight,
                        self.K4.bias,
                        self.V4.weight,
                        self.V4.bias,
                        3,stage) 
            Q = self.Q4(query)
            K = self.K4(key)
            V = self.V4(value)

            nhead = (self.nhead // 4) * 4
            Q, K, V = shape(Q, nhead), shape(K, nhead), shape(V, nhead)
            
            mode = 'mode 4'


        ## 2) Calculate scores
        Q = Q / math.sqrt(self.head_dim)
        # QK : (B, nhead, query_len, key_len)
        QK = torch.matmul(Q, K.transpose(2, 3))
        scores = QK.float()
        
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, 1 or query_len, key_len)
            scores = scores.masked_fill(mask, -1e18)
        
        ## 3) Apply dropout and compute context vector
        attn = self.softmax(scores).to(query.dtype) # (B, nhead, query_len, key_len)
        drop_attn = self.dropout(attn)

        # (B, nhead, query_len, key_len) @ (B, nhead, value_len, head_dim)

        context_original = torch.matmul(drop_attn, V)
        # (B, nhead, query_len, head_dim)

        context = unshape(context_original, nhead) # (B, q_len, d_model)
        if mode == 'mode 1':
            output = self.final_linear_1(context) # (B, q_len, d_model)
            output = self.LayerNorm(output)
        if mode == 'mode 2':
            replace_outputs(self.final_linear_2.weight, self.final_linear_2.bias, 0, stage)
            output = self.final_linear_2(context) # (B, q_len, d_model)
            output = self.LayerNorm(output)
        if mode == 'mode 3':
            replace_outputs(self.final_linear_3.weight, self.final_linear_3.bias, 1, stage)
            output = self.final_linear_3(context) # (B, q_len, d_model)
            output = self.LayerNorm(output)
        if mode == 'mode 4':
            replace_outputs(self.final_linear_4.weight, self.final_linear_4.bias, 2, stage)
            output = self.final_linear_4(context) # (B, q_len, d_model)
            output = self.LayerNorm(output)
    
        # attns = attn.view(B, self.nhead, query_len, key_len)
        return output #, attns
    
    def update_dropout(self, dropout):
        self.dropout.p = dropout
    
    # backward 계산할 때 gradient 바꿔치기
    # backward 계산 후 update 이전에 바꿔치기



if __name__=='__main__':
    max_len = 312 # 15
    mha = AutoMHA(16, 1024, 512)
    lengths = torch.LongTensor((max_len,4,3))
    mask = sequence_mask(lengths).unsqueeze(1)
    tmp = torch.randn((3,max_len,1024))
    res = mha(tmp, tmp, tmp, mask)
    classifier = nn.Softmax(-1)
    argmax = torch.max(classifier(res), -1)[1]
    idx = torch.LongTensor([2,3,4,8])
    answer = argmax.clone()
    answer[:, idx] = 0
    
    loss = torch.nn.CrossEntropyLoss()
    loss = loss(classifier(res).reshape(-1, 1024), answer.reshape(-1))
    loss.backward()
    print(res.size())
