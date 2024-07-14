import torch
import torch.nn as nn
import torch.nn.functional as F


# Classic Transformer model.
# No asserts. No robustness guarantees. Just for understanding.
# Variable names are as alinged with the original paper as possible.
# Formulas appearance is the same.

# d_model means vector dimension."we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel"


class Transformer(nn.Module):
    def __init__(self,

                 num_layers,  # Num of encoder's layers
                 d_model,    # dimension of model
                 num_heads,  # num of heads in multihead attention
                 d_ff,
                 input_vocab_size,
                 target_vocab_size):
        super().__init__()
        # self.embedding=nn.Embedding(input_vocab_size,d_model)
        # self.encoder=Encoder(num_layers,d_model,num_heads,d_ff)
        # self.output_layer=nn.Linear(d_model,target_vocab_size)

    def forward(self, src, src_mask):
        # x=self.embedding(src)
        # x=self.encoder(x,src_mask)
        # x=self.output_layer(x)
        return x


#################################################
#################################################
#                   Layers                      #
#################################################
#################################################

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_head, d_ff, dictionary_size):
        super(Encoder, self).__init__()

        self.word_embedding = nn.Embedding(dictionary_size, d_model)
        self.positonal_encoding = PositionalEncoding(d_model)
        self.stack_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_head, d_ff) for _ in range(num_layers)])

    def forward(self, x):  # no mask for encoder
        x = self.word_embedding(x)
        x = self.positonal_encoding(x)
        for layer in self.stack_layers:
            x = layer(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super().__init__()

        self.multi_head_attention = MultiHeadSelfAttentionLayer(
            d_model, num_head)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(
            d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attention_output, _ = self.multi_head_attention(x, x, x, mask)
        attention_output = self.norm1(x+attention_output)
        feedfoward_output = self.positionwise_feedforward(attention_output)
        feedfoward_output = self.norm2(attention_output+feedfoward_output)
        return feedfoward_output


#################################################
#################################################
#                  Sub-Layers                   #
#################################################
#################################################

class PositionwiseFeedForwardLayer(nn.Module):
    """ This is a simple MLP
    """

    def __init__(self,
                 d_model,
                 # dimension of feedforward hidden layer(In paper, names 'inner-layer')
                 d_ff,
                 P_dropout=0.1
                 ):

        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(P_dropout)

    def forward(self, x):
        residual = x
        x = self.W2(F.relu(self.W1(x)))  # FFN(x)=max(0,xW1+b1)W2+b2
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 h=8,             # number of head
                 P_dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.h = h
        # Multi-head's meaning is splitting d_model into h heads, and each head go through
        self.depth_q = self.depth_k = self.depth_v = d_model/h

        self.W_q = nn.Linear(d_model, d_model)  # actually [d_model, d_q*h]
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)

    def scale_dot_product_attention(self, q, k, v, mask=None):
        pass

    def forward(self, x, mask=None):

        batch_size = x.size(0)

        residual = x

        # d_q,d_k,d_v are all d_model/h in Transformer, -1 means sequence length. length of sentence.
        # for decoder, q's shape should be [batch_size, len(sentence), h,d_k]
        q = self.W_q(x).view(batch_size, -1, self.h, self.depth_k)
        k = self.W_k(x).view(batch_size, -1, self.h, self.depth_k)
        v = self.W_v(x).view(batch_size, -1, self.h, self.depth_v)
        # multi-head here is to split d_model into h heads, and each head go through the same process.
        # Questions?: imagine that we use embedding to convert a single word to a d_model vector.
        # Then, here, we separate the d_model vector into h heads. are we breaking down the features of the word?
        # and just let each query just pay attention to some features of the words?

        # Recall that torch.matmul supports batched matrix multiplication.
        # >>> # batched matrix x batched matrix
        # >> > tensor1 = torch.randn(10, 3, 4)
        # >> > tensor2 = torch.randn(10, 4, 5)
        # >>> torch.matmul(tensor1, tensor2).size()
        # torch.Size([10, 3, 5])
        attention = self.scale_dot_product_attention(q, k, v, mask)

    def ScaledDotProductAttention(self, q, k, v, mask=None):
        # q,k,v are all [batch_size,len(sentence),h,d_k]
        # first convert them to [batch_size,h,len(sentence),d_k]
        # to demostrate, assume q has shape [24,47,8,64],
        # 24 batch; 47 words; 8heads; 64 dimension for each head
        # How can we train a batch with difference length of sentence? padding in encoder input!
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-2, -1)) / \
            torch.sqrt(self.depth_k)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        attention = torch.matmul(attention, v)


#################################################
#################################################
#             Positional Encoding               #
#################################################
#################################################

class PositionalEncoding(nn.Module):
    '''
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
        row = torch.pow(10000, 2*(torch.arange(0, d_model)//2)/d_model)
        PE[:, 0::2] = torch.sin(position.unsqueeze(1)/row[0::2])
        PE[:, 1::2] = torch.cos(position.unsqueeze(1)/row[1::2])
        self.register_buffer('PE', PE)

    def forward(self, x):
        # assume x has shape of [batch_size, len(sentence), d_model]
        return x + self.PE[:x.size(1), :]


# Residual Dropout:
# We apply dropout [27] to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
# For the base model, we use a rate of Pdrop = 0.1.

# 假设句子长度是n，输入为n个长度为d_model的向量。
