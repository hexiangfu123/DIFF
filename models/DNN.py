import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from transformers import BertModel
from .transformer import make_model
import ipdb 

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, args, in_dims, out_dims, dropout=0.2):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.args = args
        self.linear1 = nn.Linear(self.in_dims, self.in_dims * 2)
        self.linear2 = nn.Linear(self.in_dims * 2, self.out_dims)
        # ipdb.set_trace()
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.uniform_(self.linear1.bias, -0.1, 0.1)
        nn.init.normal_(self.linear2.weight, std=0.01)
        nn.init.uniform_(self.linear2.bias, -0.1, 0.1)
    

    
    def forward(self, x):

        x = self.drop(x)
        h = self.linear1(x)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.linear2(h)
        h = torch.relu(h)
        # ipdb.set_trace()
        return h

class ScoreEncoder(nn.Module):
    def __init__(self, args, cat_num, dropout=0.2):
        super(ScoreEncoder, self).__init__() 
        self.args = args

        self.drop = nn.Dropout(dropout)
        # self.args.ri_emb_dim = self.args.word_dim + self.args.id_dim
        self.emb_size = (self.args.id_dim + self.args.word_dim)*cat_num
        self.result_shape = 1
        self.func = F.relu

        self.fc1 = nn.Linear(self.emb_size, 300)
        self.fc2 = nn.Linear(300, self.result_shape)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.uniform_(self.fc1.bias, a=0, b=1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc2.bias, a=0, b=1)
    
    def forward(self, x):
        # ipdb.set_trace()
        ui_feature = self.drop(x)
        ui_feature = self.func(ui_feature)
        output = self.fc1(ui_feature)
        output = self.drop(output)
        output = self.fc2(output)
        output = self.func(output)

        return output

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Transformer(nn.Module):
    def __init__(self, args, device, dropout=0.1):
        super(Transformer, self).__init__()
        self.args = args
        self.device = device
        self.time_emb_dim = args.id_dim + args.word_dim
        self.emb_dim = args.id_dim + args.word_dim
        
        if args.use_score_emb == 'cat':
            self.emb_dim += args.score_dim
        self.user_net = make_model(N=6, d_model=self.emb_dim, d_ff=self.emb_dim*4, h=8, dropout=dropout)
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.score_encoder = ScoreEncoder(self.args, 1)
        
        self.drop = nn.Dropout(0.2)


    def add_noise(self, user_emb, x_t, step_emb):
        if step_emb != None:
            x_t = x_t + step_emb
        emb_with_noise = user_emb + x_t.unsqueeze(1)

        return emb_with_noise

    def forward(self, user_emb, item_emb, x_t, step_emb, mask):
        
        if step_emb != None:
            time_emb = timestep_embedding(step_emb, self.time_emb_dim).to(item_emb.device) # batch_size * time_emb
            step_emb = self.emb_layer(time_emb)
        if item_emb != None:
            user_emb_with_noise = self.add_noise(user_emb, item_emb, step_emb).to(self.device)

        else:
            user_emb_with_noise = user_emb
        user_all_feature_cls = self.user_net(user_emb_with_noise, mask).to(self.device)
        output = torch.max(user_all_feature_cls, dim = 1).values
        return output

class VA_Atten(nn.Module):

    def __init__(self, dim1, dim2):
        super().__init__()
        self.drop_out = nn.Dropout(0.2)
        self.att_fc = nn.Linear(dim1, dim2)
        self.att_h = nn.Linear(dim2, 1)
        nn.init.xavier_uniform_(self.att_fc.weight, gain=1)
        nn.init.uniform_(self.att_h.weight, -0.1, 0.1)

    def forward(self, x, y):
        x = x + y.unsqueeze(1)
        score = self.att_h(torch.tanh(self.att_fc(x)))
        weight = F.softmax(score, -2)
        return (x*weight).sum(-2)