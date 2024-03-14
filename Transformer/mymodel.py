import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.depth=d_model//num_heads
        
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        self.linear=nn.Linear(d_model,d_model)

    def split_heads(self,x,batch_size):
        x=x.view(batch_size,-1,self.num_heads,self.depth)
        return x.permute(0,2,1,3)

    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)
        
        query=self.W_q(query)
        key=self.W_k(key)
        value-self.W_v(value)
        
        query=self.split_heads(query,batch_size)
        key=self.split_heads(key,batch_size)
        value=self.split_heads(value,batch_size)
        
        scaled_dot_product_attention_logits=torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(self.depth,dtype=torch.float32))
        
        if mask is not None:
            scaled_dot_product_attention_logits+=(mask*-1e9)
            
        attention_weights=F.softmax(scaled_dot_product_attention_logits,dim=-1)
        output=torch.matmul(attention_weights,value)
        
        output=output.permute(0,2,1,3).contiguous().view(batch_size,-1,self.d_model)
        output=self.linear(output)
        
        return output, attention_weights

class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_head,d_ff):
        super().__init__()
        self.multi_head_attention=MultiHeadSelfAttentionLayer(d_model,num_head) # TODO:
        self.positionwise_feedforward=PositionwiseFeedForwardLayer(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self,x,mask):
        attention_output,_=self.multi_head_attention(x,x,x,mask)
        attention_output=self.norm1(x+attention_output)
        feedfoward_output=self.positionwise_feedforward(attention_output)
        feedfoward_output=self.norm2(attention_output+feedfoward_output )
        return feedfoward_output
        
        

class Encoder(nn.Module):
    def __init__(self,num_layers,d_model,num_head,d_ff):
        super(Encoder,self).__init__()
        self.layers=nn.ModuleList([EncoderLayer(d_model,num_head,d_ff) for _ in range(num_layers)])
    
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return x
    
    
class Transformer(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,d_ff,input_vocab_size,target_vocab_size):
        super().__init__()
        
        self.embedding=nn.Embedding(input_vocab_size,d_model)
        self.encoder=Encoder(num_layers,d_model,num_heads,d_ff)
        self.output_layer=nn.Linear(d_model,target_vocab_size)
        
    def forward(self,src,src_mask):
        x=self.embedding(src)
        x=self.encoder(x,src_mask)
        x=self.output_layer(x)
        return x