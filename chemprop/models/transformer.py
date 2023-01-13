import math
import torch
import copy
import torch.nn as nn
from chemprop.args import TrainArgs
import pickle, os
import numpy as np

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, attn_bias = None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    
    attn_bias = attn_bias.permute(0, 3, 1, 2) #8x30x30x3 -> 8x3x30x30

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #8x3x30x30
    scores = scores + attn_bias

    if mask is not None: #Mask: 8x1x1x30
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1) #(B, 3, 30, 30)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), scores

class MultiHeadedAttention(nn.Module):
    def __init__(self, args: TrainArgs, d_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % args.transformer_heads == 0
        
        self.d_k = d_model // args.transformer_heads
        self.h = args.transformer_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=args.dropout)

        self.print_attention = args.print_attention

    def forward(self, query, key, value, attn_bias=None, mask=None, is_last_layer = False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ] #1x3x30x100

        # print(self.linears[0].weight)
        # print(self.linears[1].weight)

        # with open('TEST_query_proj.pkl', 'wb') as f:
        #     pickle.dump(self.linears[0].weight, f)
        # with open("TEST_key_proj.pkl", 'wb') as f:
        #     pickle.dump(self.linears[1].weight, f)
        # print("attn bias in attention", attn_bias.shape)
        # with open("TEST_attn_bias.pkl", 'wb') as f:
        #     pickle.dump(attn_bias, f)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, attn_bias=attn_bias, mask=mask, dropout=self.dropout
        )
        # p_attn = self.attn.softmax(dim=-1)
        # a = p_attn[:,:,0,1:].sum(dim=-1)
        # b = p_attn[0,0,1:9,:9]
        # print(is_last_layer)
        # print("cls", a)
        # print(b)
        # raise Exception

        # zero_attn_bool = torch.isclose(a, torch.zeros(1).to(p_attn.device)).any()
        # if zero_attn_bool:
        # #     print(a)
        #     print(zero_attn_bool)
        #     raise Exception
        #Quite bad but whatever
        if (self.print_attention and is_last_layer and self.training == False):
        # if (self.print_attention and self.training == False):
            if os.path.exists('attention_weights.pkl'):
                with open('attention_weights.pkl', 'rb') as f:
                    weights = pickle.load(f)
            else:
                weights = []

            weights.extend(self.attn.tolist())

            with open('attention_weights.pkl', 'wb') as f:
                pickle.dump(weights, f)
        
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    # def __init__(self, layer, N, num_kernels, num_heads):
    def __init__(self, layer, args: TrainArgs):
        super(Encoder, self).__init__()

        self.N = args.transformer_layers
        num_kernels = args.num_kernels
        num_heads = args.transformer_heads

        self.layers = clones(layer, self.N)
        self.norm = LayerNorm(layer.size)

        self.gbf_projection = GaussianProjection(num_kernels, num_heads)
        self.w0 = nn.Parameter(torch.zeros(1,1,1,3))

    def forward(self, x, mask, spatial_features):
        attn_bias = self.gbf_projection(spatial_features) * self.w0

        # with open("TEST_x.pkl", "wb") as f:
        #     pickle.dump(x, f)

        # print("attnbias", attn_bias.shape)
        # with torch.no_grad():
        #     prev = x[:, 0, :].detach().cpu().numpy()
        for i, layer in enumerate(self.layers):
            is_last_layer = (i == self.N - 1)
            x = layer(x, mask, attn_bias, is_last_layer)
        
        # with torch.no_grad():
        #     delta = np.absolute(prev - x[:, 0, :].detach().cpu().numpy()).mean()
        return self.norm(x) #, delta

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    # def __init__(self, size, num_heads, size_ff, dropout):
    def __init__(self, args: TrainArgs):
        super(EncoderLayer, self).__init__()

        size = args.hidden_size
        # if args.use_spatial_features:
        #     size += args.num_kernels
        
        dropout = args.dropout
        size_ff = args.ffn_hidden_size

        self.self_attn = MultiHeadedAttention(args, size)
        self.feed_forward = PositionwiseFeedForward(size, size_ff, dropout)

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, attn_bias, is_last_layer):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_bias, mask, is_last_layer)) # x -> norm -> self attention -> dropout
        return self.sublayer[1](x, self.feed_forward) # x -> norm -> feed forward -> dropout

def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.K = args.num_kernels
        self.edge_types = args.edge_types 

        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        self.mul = nn.Embedding(self.edge_types, 1, padding_idx=0) 
        self.bias = nn.Embedding(self.edge_types, 1, padding_idx=0) 
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types) #For each embeddings, get features
        bias = self.bias(edge_types)

        x = mul * x.unsqueeze(-1) + bias #8x30x30x1
        x = x.expand(-1, -1, -1, self.K) #batch * max_atoms * max_atoms * kernels (8x30x30x16)

        mean = self.means.weight.float().view(-1) #k
        std = self.stds.weight.float().view(-1).abs() + 1e-2 #k

        return gaussian(x.float(), mean, std) #8x30x30x16

class GaussianProjection(nn.Module):
    def __init__(self, K=60, output_size=3):
        super(GaussianProjection, self).__init__()

        self.layer1 = nn.Linear(K, K)
        self.layer2 = nn.Linear(K, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.gelu(x)
        x = self.layer2(x)
        return x
