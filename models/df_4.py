import enum
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import ipdb
import models.gaussian_diffusion as gd
from models.DNN import DNN, ScoreEncoder, Transformer, VA_Atten
import argparse

class df_4(nn.Module):
    def __init__(self, mean_type, args, device, n_item, history_num_per_term=10, beta_fixed=True, time_type="cat"):
        super(df_4, self).__init__()
        self.args = args
        self.device = device
        self.mask_item_num = args.item_num - 1
        self.cluster = self.args.cluster
        self.diffusion1 = gd.GaussianDiffusion(mean_type, self.args.noise_schedule, \
            self.args.noise_scale, self.args.noise_min, self.args.noise_max, self.args.steps, device).to(device)
        self.score_encoder = ScoreEncoder(self.args, 2).to(device)
        self.item_id_emb = nn.Embedding(self.args.item_num, self.args.id_dim).to(device)
        self.user_id_emb = nn.Embedding(self.args.user_num, self.args.id_dim).to(device)
        self.transformer =  Transformer(self.args, device).to(device)
        self.va_atten = VA_Atten(self.args.emb_size, 400)
        self.dnn1 = DNN(self.args, self.args.emb_size*3, self.args.emb_size).to(device)
        self.score_enc = nn.Parameter(torch.FloatTensor([5]*self.args.i_max_r), requires_grad=True)

    def forward(self, x, target, reweight, epoch):
        # ================= review & item id embedding =================
        (a, b) = x
        user_reviews_emb, item_reviews_emb, ui_reviews_emb = a
        uids, iids, user_item2id, item_user2id, scores = b
        iids_emb = self.item_id_emb(iids)  # [batch_size * id_dim]
        uids_emb = self.user_id_emb(uids)
        user_item2id_emb = self.item_id_emb(user_item2id) # [batch_size * user_review_num * id_dim]
        item_user2id_emb = self.user_id_emb(item_user2id) # [batch_size * user_review_num * id_dim]
        mask = torch.where(user_item2id==self.mask_item_num, 0., 1.) + torch.where(user_item2id==iids.unsqueeze(1), 0., 1.)
        mask = torch.where(mask == 2, 1., 0.)

        # ================= review & item id & score cat =================

        review_emb = torch.cat((iids_emb, ui_reviews_emb), 1)
        user_emb = torch.cat((user_item2id_emb, user_reviews_emb), 2)
        item_emb = torch.cat((item_user2id_emb, item_reviews_emb), 2)
        item_emb_max = torch.max(item_emb, dim = 1).values
        user_emb_max = torch.max(user_emb, dim = 1).values

        loss = {}
        # ================= diffusion training =================
        if False:
            output = self.transformer(user_emb, item_emb, None, None, mask)
            output = torch.cat((user_emb, output), 1)
        else:
            review_emb_pre = torch.cat((user_emb_max, item_emb_max, review_emb), dim=1)
            review_emb_pre = self.dnn1(review_emb_pre)
            review_emb_pre = review_emb_pre + review_emb * torch.randn_like(review_emb)

            out = self.diffusion1.training_losses(self.transformer, review_emb, user_emb, mask, reweight=True)
            output = out["pred_xstart"]
            loss['diffusion'] = out["loss"]
            output1 = torch.softmax(item_emb.matmul(output.unsqueeze(-1)).squeeze(-1), 1)
        scores = torch.sum(output1 * scores, dim=1)

        loss['mse_loss'] = (target - scores)**2
        loss['mae_loss'] = abs(target - scores)
        loss['adj_reverse_loss'] = (review_emb_pre - review_emb)**2
        return loss
        

    def predict(self, x, sampling_steps, sampling_noise):
        # ================= review & item id embedding =================
        (a, b) = x
        user_reviews_emb, item_reviews_emb, ui_reviews_emb = a
        uids, iids, user_item2id, item_user2id, scores = b
        # ipdb.set_trace()
        iids_emb = self.item_id_emb(iids)  # [batch_size * id_dim]
        uids_emb = self.user_id_emb(uids)
        # ipdb.set_trace()
        user_item2id_emb = self.item_id_emb(user_item2id) # [batch_size * user_review_num * id_dim]
        item_user2id_emb = self.user_id_emb(item_user2id) # [batch_size * user_review_num * id_dim]

        mask = torch.where(user_item2id==self.mask_item_num, 0., 1.) + torch.where(user_item2id==iids.unsqueeze(1), 0., 1.)
        mask = torch.where(mask == 2, 1., 0.)

        # ================= review & item id & score cat =================

        review_emb = torch.cat((iids_emb, ui_reviews_emb), 1)
        user_emb = torch.cat((user_item2id_emb, user_reviews_emb), 2)
        item_emb = torch.cat((item_user2id_emb, item_reviews_emb), 2)
        user_emb_max = torch.max(user_emb, dim = 1).values
        item_emb_max = torch.max(item_emb, dim = 1).values

        # ================= diffusion reverse =================          
        if True:
            review_emb_pre = self.transformer(user_emb, item_emb_max, None, None, mask)
            out = self.diffusion1.p_sample(self.transformer, review_emb_pre, user_emb, mask, self.args.sampling_steps, self.args.sampling_noise)
            output1 = torch.softmax(item_emb.matmul(out.unsqueeze(-1)).squeeze(-1), 1)
        else:
            out = self.transformer(user_emb, item_emb, None, None, mask)
            out = torch.cat((item_emb, review_emb), 1)
        scores = torch.sum(output1 * scores, dim=1)
        return scores
    
    
    def init_word_emb(self):
        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.word_embs.weight)

        nn.init.xavier_normal_(self.user_id_emb.weight)
        nn.init.xavier_normal_(self.item_id_emb.weight)

