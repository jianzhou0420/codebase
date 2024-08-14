import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
# Classic Transformer model.
# No asserts. No robustness guarantees. Just for understanding.
# Variable names are as alinged with the original paper as possible.
# Formulas appearance is the same.

# d_model means vector dimension."we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel"


class Transformer(nn.Module):
    def __init__(self, config: yaml):
        super().__init__()
        # self.embedding=nn.Embedding(input_vocab_size,d_model)
        # self.encoder=Encoder(num_layers,d_model,num_heads,d_ff)
        # self.output_layer=nn.Linear(d_model,target_vocab_size)

        # fetch local variables
        d_model = config['d_model']
        target_vocab_size = config['target_vocab_size']

        # defind layers
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.d_model2target = nn.Linear(d_model, target_vocab_size, bias=False)

    def forward(self, src_seq, target_seq, src_mask, target_mask):

        output_encoder = self.encoder(src_seq, src_mask)
        output_decoder = self.decoder(target_seq, output_encoder, target_mask, src_mask)

        target_seq = self.d_model2target(output_decoder)

        return target_seq


#################################################
#################################################
#                   Modules                      #
#################################################
#################################################


class Encoder(nn.Module):
    def __init__(self, config: yaml):
        super(Encoder, self).__init__()

        # Fetch local variables
        num_layers = config['num_layers']
        d_model = config['d_model']
        dictionary_size = config['dictionary_size']
        dropout = config['dropout']
        # Define components
        self.word_embedding = nn.Embedding(dictionary_size, d_model)
        self.positonal_encoding = PositionalEncoding(d_model)
        self.stack_layers = nn.ModuleList([EncoderLayer(config) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x has shape [batch_size, len(sentence)]
        x = self.word_embedding(x)
        x = self.positonal_encoding(x)
        x = self.dropout(x)
        # 在github上，比较出名的attention is all you need的实现，这里是有一个layer_norm(PE_output)的，不过原文里没有，我也就没有加了。

        for layer in self.stack_layers:
            x = layer(x, mask=mask)
        return x


class Decoder(nn.Module):
    def __init__(self, config: yaml):
        super(Decoder, self).__init__()

        # Fetch local variables
        d_model = config['d_model']
        self.num_layer = num_layer = config['num_layer']
        tar_vocab_size = config['target_vocab_size']
        P_dropout = config['P_dropout']
        # Define components
        self.target_word_embedding = nn.Embedding(tar_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(P_dropout)
        self.stack_layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tar_seq, encode_output, self_attn_mask, decode_attn_mask):
        x = tar_seq
        # embedding
        x = self.target_word_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        # 在github上，比较出名的attention is all you need的实现，这里是有一个layer_norm(PE_output)的，不过原文里没有，我也就没有加了。

        for layer in self.num_layer:
            x = layer(x, encode_output, self_attn_mask, decode_attn_mask)

        return x


#################################################
#################################################
#                   Layers                      #
#################################################
#################################################

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Fetch local variables
        d_model = config['d_model']

        # Define components
        self.multi_head_attention = MultiHeadAttentionLayer(config)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attention_output = self.multi_head_attention(x, x, x, mask)
        add_and_norm1 = self.norm1(x + attention_output)

        feedfoward_output = self.positionwise_feedforward(add_and_norm1)
        add_and_norm2 = self.norm2(attention_output + feedfoward_output)

        return add_and_norm2


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        # Fetch local variables
        d_model = config['d_model']

        # Define components
        self.self_attention = MultiHeadAttentionLayer(config)
        self.cross_attention = MultiHeadAttentionLayer(config)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(config)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, decode_input, encode_output, self_attn_mask, decode_attn_mask):

        self_attention_output = self.self_attention(decode_input, decode_input, decode_input, self_attn_mask)
        add_and_norm1 = self.norm1(decode_input + self_attention_output)

        decode_attention_output = self.cross_attention(add_and_norm1, encode_output, encode_output, decode_attn_mask)
        add_and_norm2 = self.norm2(add_and_norm1 + decode_attention_output)

        decode_output = self.positionwise_feedforward(add_and_norm2)
        add_and_norm3 = self.norm3(add_and_norm2 + decode_output)

        return add_and_norm3
#################################################
#################################################
#                  Sub-Layers                   #
#################################################
#################################################


class PositionwiseFeedForwardLayer(nn.Module):
    """ This is a simple MLP
    """

    def __init__(self, config: yaml):
        super().__init__()

        # Fetch local variables
        d_model = config['d_model']
        d_ff = config['d_ff']
        P_dropout = config['P_dropout']

        # Define components
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(P_dropout)

    def forward(self, x):
        x = self.W2(F.relu(self.W1(x)))  # FFN(x)=max(0,xW1+b1)W2+b2
        x = self.dropout(x)              # Dropout as in section 5.4
        # Add&Norm is in EncoderLayer scope
        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, config: yaml):
        super().__init__()

        # Fetch local variables
        self.d_model = d_model = config['d_model']
        h = config['num_heads']
        P_dropout = config['P_dropout']
        self.depth_k = self.depth_v = d_model / h   # while, I think self.depth_q should be d_model/h as well. so that y can just make it depth=d_model/h

        # Define components
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(P_dropout)

    def forward(self, x_q, x_k, x_v, mask=None):  # when self-attention, x_q=x_k=x_v. when cross-attention, x_k=x_v but x_q is different.
        batch_size = x_q.size(0)

        q = self.W_q(x_q).view(batch_size, -1, self.h, self.depth_k)  # follow the paper, instead I think it should be self.depth_q,but actually they are the same
        k = self.W_k(x_k).view(batch_size, -1, self.h, self.depth_k)
        v = self.W_v(x_v).view(batch_size, -1, self.h, self.depth_v)

        attention = self.scale_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # concat heads & linear
        attention = self.linear(attention)
        attention = self.dropout(attention)  # Dropout as in section 5.4
        # Add&Norm is in EncoderLayer scope
        return attention

    def scale_dot_product_attention(self, q, k, v, mask=None):
        # q,k,v are all [batch_size,len(sentence),h,d_k]
        # first convert them to [batch_size,h,len(sentence),d_k]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(self.depth_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # for heads boardcasting
            qk = qk.masked_fill(mask == 0, -1e9)

        attention = torch.matmul(F.softmax(qk, dim=-1), v)

        return attention


#################################################
#################################################
#             Positional Encoding               #
#################################################
#################################################


class PositionalEncoding(nn.Module):
    '''
    1D PE: Sequence PE
    Assume we have embedding a sentence to shape [len(sentence), d_model] where d_model is the dimension of embedding. one world one embedding vector.
    embedding vector has d_model dimension.
    Positional encoding is to use a formula to add some values to each dimension of embedding vector. The specific value to add on each element of embedding vector is determined by the position of the element in the sentence.
    depends on the position of the word and the dimension of the embedding.

    '''

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # we are goona create a PE table. It is a tensor with shape [max_len, d_model]
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        row = torch.pow(10000, 2 * (torch.arange(0, d_model) // 2) / d_model)
        PE[:, 0::2] = torch.sin(position.unsqueeze(1) / row[0::2])
        PE[:, 1::2] = torch.cos(position.unsqueeze(1) / row[1::2])
        self.register_buffer('PE', PE)

    def forward(self, x):
        # assume x has shape of [batch_size, len(sentence), d_model]
        return x + self.PE[:x.size(1), :]


# Residual Dropout:
# We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# For the base model, we use a rate of Pdrop = 0.1.

# 假设句子长度是n，输入为n个长度为d_model的向量。
# 不对，就算你是self-attention


def get_pad_mask():
    pass


def get_subsequent_mask():
    pass


if __name__ == '__main__':
    with open('transformer_args.yaml', 'r') as file:
        config = yaml.safe_load(file)

    instance_transformer = Transformer(config)
