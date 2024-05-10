import fire
import argparse
from ast import parse
import os
import time
import numpy as np
import copy
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import config
import models.gaussian_diffusion as gd
from models.DNN import DNN
from models.df_4 import df_4 as df_4
from dataset import AmazonData
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def collate_fn(batch):
    # np.random.seed(random_seed + worker_id)
    data, label = zip(*batch)
    return data, label
 

def unpack_input(opt, x, score_matrix):
    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.user_list[uids] 
    item_reviews = opt.item_list[iids] 
    user_item2id = opt.user2itemid_dict[uids] 
    item_user2id = opt.item2userid_dict[iids]
    user_item2review = opt.user_item2review.item()
    reviews = []
    scores = []
    # ipdb.set_trace()
    for i in range(len(x)):
        iid = iids[i]
        uid = uids[i]
        u2iids = item_user2id[i]
        li = []
        for x in u2iids:
            if x != uid:
                try:
                    li.append(score_matrix[x][iid])
                except:
                    li.append(0)
            else:
                li.append(0)
        scores.append(li)
        review = user_item2review[uid][iid]
        if opt.use_bert_emb:
            review = review.view(-1).tolist()
        reviews.append(review)
      
    train_data = [user_reviews, item_reviews, reviews]
    train_data = list(map(lambda x: torch.FloatTensor(x).cuda(), train_data))
    train_data_int = [uids, iids, user_item2id, item_user2id, scores]
    train_data_int = list(map(lambda x: torch.LongTensor(x).cuda(), train_data_int))
    return (train_data, train_data_int)

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def train(**kwargs):
    if 'dataset' not in kwargs:
        args = getattr(config, 'Toys_and_Games_data_Config')()
    else:
        args = getattr(config, kwargs['dataset'] + '_data_Config')()
    args.parse(kwargs)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###


    train_data = AmazonData(args.data_root, train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    test_data = AmazonData(args.data_root, train=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print('{}: train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))
    # ipdb.set_trace()
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print('data ready.')


    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    if args.use_bert_emb:
        args.word_dim = 768
        args.id_dim = 64
        args.emb_size = args.word_dim +  args.id_dim

    model = df_4(mean_type, args, device, args.item_num).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    save_path = f"./saved_models/{args.dataset}_diffusion_model_with_{args.steps}step.pth"
    print("models ready.")

    def evaluate(data_loader, model=model):
        model.eval()
        loss_mae = 0.0
        loss_mse = 0.0
        total = 0
        with torch.no_grad():
            for batch_idx, (batch, target) in enumerate(data_loader):
                input1 = unpack_input(args, batch, train_data.score_matrix)
                target = torch.LongTensor(target).to(device)
                prediction = model.predict(input1, args.sampling_steps, args.sampling_noise)
            
                loss_mae += torch.sum(abs(target-prediction)).item()
                loss_mse += torch.sum(((target-prediction)**2)).item()
                total += target.shape[0]
            
        return loss_mae/test_data_size, loss_mse/test_data_size

    best_mse, best_epoch = 100000, 0

    print("Start training...")
    for epoch in range(1, args.epochs + 1):

        model.train()
        total_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        print("{} Epoch {}: start".format(now(), epoch))
        for batch_idx, (batch, target) in enumerate(train_loader):
            input1 = unpack_input(args, batch, train_data.score_matrix)
            optimizer.zero_grad()
            target = torch.LongTensor(target).to(device)
            
            loss = model(input1, target, args.reweight, epoch)
            train_mse += torch.sum(loss['mse_loss'])
            train_mae += torch.sum(loss['mae_loss'])
            loss = loss['mse_loss'].mean() + loss['diffusion'].mean()

            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print('Total loss: {}'.format(abs(total_loss)))
        train_mse = train_mse / train_data_size
        train_mae = train_mae / train_data_size
        print("{};train reslut: mse: {}; rmse: {}; mae: {}".format(now(), train_mse, math.sqrt(train_mse), train_mae))

        mae, mse = evaluate(test_loader)
        print("{};test reslut: mse: {}; rmse: {}; mae: {}".format(now(), mse, math.sqrt(mse), mae))
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print("----"*20)
    print(f"{now()} {args.dataset} epoch:  {best_epoch}  best_res:  {best_mse}")
    print("----"*20)

if __name__ == "__main__":
    fire.Fire()



