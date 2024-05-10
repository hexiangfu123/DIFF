# -*- coding: utf-8 -*-
import numpy as np
import os
import ipdb
import torch

class DefaultConfig:
    dataset = 'Digital_Music'

    data_path = './dataset/'
    lr = 1e-5
    alpha = 0.2
    cat_input = True
    no_diff = False
    use_bert_emb = True
    use_score_emb = 'none' # 'cat', 'add', 'none'
    split_diff_score = False
    weight_decay = 0.0
    # batch_size = 400
    cuda = True
    gpu = '1'
    num_workers = 20
    epochs = 50
    save_path = './saved_models/'
    log_name = 'log'
    round = 1
    saved = False
    loss = 'mlp'

    # --------------Autoencoder---------------------#
    n_cate = 1
    in_dims = [300]
    out_dims = []
    act_func = 'tanh'
    lamda = 0.03
    optimizer1 = 'AdamW'
    anneal_cap = 0.005
    anneal_steps = 500
    vae_anneal_cap = 0.3
    vae_anneal_steps = 200
    reparam = True
    # --------------main model---------------------#
    cluster = 2
    time_type = 'cat'
    dims = '[2]'
    norm = False
    emb_size = 10
    word_dim = 300
    score_dim = 32
    id_dim = 60
    # ri_emb_dim = 200
    # --------------diffusion---------------------#
    mean_type = 'x0'
    steps = 80
    noise_schedule = 'linear-var'
    noise_scale = 0.1
    noise_min = 0.0001
    noise_max = 0.02
    sampling_noise = True
    sampling_steps = 0
    reweight = True

    use_word_embedding = True

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        # self.user2review = np.load(self.user2review_path, encoding='bytes', allow_pickle=True)
        # self.item2review = np.load(self.item2review_path, encoding='bytes', allow_pickle=True)

        
        # self.item_list = np.load(self.item_list_path, encoding='bytes', allow_pickle=True)
        self.user2itemid_dict = np.load(self.user2itemid_path, encoding='bytes', allow_pickle=True)
        self.item2userid_dict = np.load(self.item2userid_path, encoding='bytes', allow_pickle=True)
        if self.use_bert_emb:
            # ipdb.set_trace()
            self.user_item2review = np.load(self.ui2r_path, encoding='bytes', allow_pickle=True)
            if not os.path.exists(self.u2ir_path):
                save_user_list(self)
            if not os.path.exists(self.i2ur_path):
                save_item_list(self)
            # ipdb.set_trace()
            self.user_list = np.load(self.u2ir_path, encoding='bytes', allow_pickle=True)
            self.item_list = np.load(self.i2ur_path, encoding='bytes', allow_pickle=True)
        else:
            self.user_list = np.load(self.user_list_path, encoding='bytes', allow_pickle=True)
            self.item_list = np.load(self.item_list_path, encoding='bytes', allow_pickle=True)
            self.user_item2review = np.load(self.item_user2Review_path, encoding='bytes', allow_pickle=True)
        


        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class Office_Products_data_Config(DefaultConfig):
    data_root = './dataset/Office_Products_data/'
    w2v_path = './dataset/Office_Products_data/train/npy/w2v.npy'

    user_list_path = './dataset/Office_Products_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Office_Products_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Office_Products_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Office_Products_data/train/npy/item_user2id.npy'
    item_user2Review_path = './dataset/Office_Products_data/train/npy/user_item2Review.npy'
    # user2review_path = './dataset/Office_Products_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Office_Products_data/train/npy/item2review.npy'
    ui2r_path = './dataset/Office_Products_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Office_Products_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Office_Products_data/train/npy/i2ur.npy'
    vocab_size = 46928
    word_dim = 300
    r_max_len = 248  # review max length

    train_data_size = 42611
    test_data_size = 10647

    user_num = 4905 + 2
    item_num = 2420 + 2

    u_max_r = 14
    i_max_r = 35

    user_mlp = [500, 80]
    item_mlp = [500, 80]

    batch_size = 128
    print_step = 200


class Gourmet_Food_data_Config(DefaultConfig):
    data_root = './dataset/Gourmet_Food_data/'
    w2v_path = './dataset/Gourmet_Food_data/train/npy/w2v.npy'

    user_list_path = './dataset/Gourmet_Food_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Gourmet_Food_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Gourmet_Food_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Gourmet_Food_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Gourmet_Food_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 74572
    word_dim = 300
    r_max_len = 168  # review max length

    u_max_r = 15
    i_max_r = 22

    train_data_size = 121003
    test_data_size = 15125
    user_num = 14681 + 2
    item_num = 8713 + 2
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 64
    print_step = 1000


class Video_Games_data_Config(DefaultConfig):
    data_root = './dataset/Video_Games_data/'
    # w2v_path = './dataset/Video_Games_data/train/npy/w2v.npy'

    # user_list_path = './dataset/Video_Games_data/train/npy/userReview2Index.npy'
    # item_list_path = './dataset/Video_Games_data/train/npy/itemReview2Index.npy'

    # user2itemid_path = './dataset/Video_Games_data/train/npy/user_item2id.npy'
    # item2userid_path = './dataset/Video_Games_data/train/npy/item_user2id.npy'
    user2review_path = './dataset/Video_Games_data/train/npy/user2review.npy'
    item2review_path = './dataset/Video_Games_data/train/npy/item2review.npy'

    vocab_size = 169398
    word_dim = 300
    r_max_len = 394  # review max length

    train_data_size = 185439
    test_data_size = 23170
    user_num = 24303 + 2
    item_num = 10672 + 2
    u_max_r = 10
    i_max_r = 27
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 8
    print_step = 1000


class Toys_and_Games_data_Config(DefaultConfig):
    data_root = './dataset/Toys_and_Games_data'
    w2v_path = './dataset/Toys_and_Games_data/train/npy/w2v.npy'

    user_list_path = './dataset/Toys_and_Games_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Toys_and_Games_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Toys_and_Games_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Toys_and_Games_data/train/npy/item_user2id.npy'
    # user2review_path = './dataset/Toys_and_Games_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Toys_and_Games_data/train/npy/item2review.npy'
    item_user2Review_path = './dataset/Toys_and_Games_data/train/npy/user_item2Review.npy'
    ui2r_path = './dataset/Toys_and_Games_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Toys_and_Games_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Toys_and_Games_data/train/npy/i2ur.npy'

    vocab_size = 77076
    word_dim = 300

    r_max_len = 178  # review max length

    train_data_size = 134081
    test_data_size = 16758
    user_num = 19412 + 2
    item_num = 11924 + 2
    u_max_r = 9
    i_max_r = 18
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000


class Kindle_Store_data_Config(DefaultConfig):
    data_root = './dataset/Kindle_Store_data/'
    w2v_path = './dataset/Kindle_Store_data/train/npy/w2v.npy'

    user_list_path = './dataset/Kindle_Store_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Kindle_Store_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Kindle_Store_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Kindle_Store_data/train/npy/item_user2id.npy'

    item_user2Review_path = './dataset/Kindle_Store_data/train/npy/user_item2Review.npy'
    ui2r_path = './dataset/Kindle_Store_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Kindle_Store_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Kindle_Store_data/train/npy/i2ur.npy'
    # ratingMatrix_path = './dataset/Kindle_Store_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 278914
    word_dim = 300

    r_max_len = 211  # review max length

    train_data_size = 786159
    test_data_size = 98230
    user_num = 68223 + 2
    item_num = 61934 + 2
    u_max_r = 20
    i_max_r = 24
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 32
    print_step = 1000


class Movies_and_TV_data_Config(DefaultConfig):
    data_root = './dataset/Movies_and_TV_data/'
    w2v_path = './dataset/Movies_and_TV_data/train/npy/w2v.npy'

    user_list_path = './dataset/Movies_and_TV_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Movies_and_TV_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Movies_and_TV_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Movies_and_TV_data/train/npy/item_user2id.npy'

    ratingMatrix_path = './dataset/Movies_and_TV_data/matrix/npy/ratingMatrix.npy'

    vocab_size = 764339
    word_dim = 300

    r_max_len = 326  # review max length

    train_data_size = 1358101
    test_data_size = 169716
    user_num = 123960
    item_num = 50052
    u_max_r = 16
    i_max_r = 49
    user_mlp = [1000, 80]
    item_mlp = [2000, 80]
    batch_size = 16
    print_step = 5000


class Clothing_Shoes_and_Jewelry_data_Config(DefaultConfig):
    data_root = './dataset/Clothing_Shoes_and_Jewelry_data/'
    w2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/w2v.npy'
    id2v_path = './dataset/Clothing_Shoes_and_Jewelry_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Clothing_Shoes_and_Jewelry_data/train/npy/itemSummary2Index.npy'
    vocab_size = 67812
    word_dim = 300
    r_max_len = 97  # review max length
    s_max_len = 31  # summary max length

    train_data_size = 222984
    test_data_size = 55693
    user_num = 39387
    item_num = 23033
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class Sports_and_Outdoors_data_Config(DefaultConfig):
    data_root = './dataset/Sports_and_Outdoors_data/'
    w2v_path = './dataset/Sports_and_Outdoors_data/train/npy/w2v.npy'
    id2v_path = './dataset/Sports_and_Outdoors_data/matrix/npy/rawMatrix.npy'
    user_list_path = './dataset/Sports_and_Outdoors_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Sports_and_Outdoors_data/train/npy/itemReview2Index.npy'
    user_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/userSummary2Index.npy'
    item_summary_path = './dataset/Sports_and_Outdoors_data/train/npy/itemSummary2Index.npy'
    vocab_size = 100129
    word_dim = 300
    r_max_len = 146  # review max length
    s_max_len = 29  # summary max length

    train_data_size = 237095
    test_data_size = 59242
    user_num = 35598
    item_num = 18357
    user_mlp = [2000, 80]
    item_mlp = [4000, 80]
    batch_size = 80
    print_step = 1000


class yelp2013_data_Config(DefaultConfig):
    data_root = './dataset/yelp2013_data'
    w2v_path = './dataset/yelp2013_data/train/npy/w2v.npy'

    user_list_path = './dataset/yelp2013_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/yelp2013_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/yelp2013_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/yelp2013_data/train/npy/item_user2id.npy'

    vocab_size = 65751
    word_dim = 300

    r_max_len = 271

    u_max_r = 57
    i_max_r = 59

    train_data_size = 63172
    test_data_size = 7897

    user_num = 1631 + 2
    item_num = 1633 + 2

    batch_size = 32
    print_step = 1000


class yelp2014_data_Config(DefaultConfig):
    data_root = './dataset/yelp2014_data'
    w2v_path = './dataset/yelp2014_data/train/npy/w2v.npy'

    user_list_path = './dataset/yelp2014_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/yelp2014_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/yelp2014_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/yelp2014_data/train/npy/item_user2id.npy'

    vocab_size = 113166
    word_dim = 300

    r_max_len = 284

    u_max_r = 58
    i_max_r = 68

    train_data_size = 184930
    test_data_size = 23116

    user_num = 4818 + 2
    item_num = 4194 + 2

    batch_size = 32
    print_step = 1000


class Musical_Instruments_data_Config(DefaultConfig):
    data_root = './dataset/Musical_Instruments_data'
    w2v_path = './dataset/Musical_Instruments_data/train/npy/w2v.npy'

    user_list_path = './dataset/Musical_Instruments_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Musical_Instruments_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Musical_Instruments_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Musical_Instruments_data/train/npy/item_user2id.npy'
    # user2review_path = './dataset/Musical_Instruments_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Musical_Instruments_data/train/npy/item2review.npy'
    item_user2Review_path = './dataset/Musical_Instruments_data/train/npy/user_item2Review.npy'
    ui2r_path = './dataset/Musical_Instruments_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Musical_Instruments_data/train/npy/u2ir.npy'
    vocab_size = 19054

    r_max_len = 154

    u_max_r = 9
    i_max_r = 13

    train_data_size = 8214
    test_data_size = 2047

    user_num = 1428 + 2
    item_num = 899 + 2

    batch_size = 64

class Amazon_data_Config(DefaultConfig):
    data_root = './dataset/Amazon_data'
    # w2v_path = './dataset/yelp2016_data/train/npy/w2v.npy'

    # user_list_path = './dataset/yelp2016_data/train/npy/userReview2Index.npy'
    # item_list_path = './dataset/yelp2016_data/train/npy/itemReview2Index.npy'

    # user2itemid_path = './dataset/yelp2016_data/train/npy/user_item2id.npy'
    # item2userid_path = './dataset/yelp2016_data/train/npy/item_user2id.npy'
    user2review_path = './dataset/Amazon_data/train/npy/user2review.npy'
    # np.load('user2review.npy', allow_pickle=True)
    item2review_path = './dataset/Amazon_data/train/npy/item2review.npy'
    vocab_size = 264045
    word_dim = 300

    r_max_len = 189

    u_max_r = 512
    i_max_r = 512
    u_avg_r = 1770
    i_avg_r = 3105

    train_data_size = 395601
    test_data_size = 98880

    user_num = 48729
    item_num = 27777

    batch_size = 8

class Digital_Music_data_Config(DefaultConfig):
    data_root = './dataset/Digital_Music_data'
    w2v_path = './dataset/Digital_Music_data/train/npy/w2v.npy'

    user_list_path = './dataset/Digital_Music_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Digital_Music_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Digital_Music_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Digital_Music_data/train/npy/item_user2id.npy'
    # user2review_path = './dataset/Digital_Music_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Digital_Music_data/train/npy/item2review.npy'
    item_user2Review_path = './dataset/Digital_Music_data/train/npy/user_item2Review.npy'

    ui2r_path = './dataset/Digital_Music_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Digital_Music_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Digital_Music_data/train/npy/i2ur.npy'
    vocab_size = 107680
    word_dim = 300

    r_max_len = 365

    u_max_r = 16
    i_max_r = 31

    train_data_size = 51764
    test_data_size = 12942

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 64


class yelp2016_data_Config(DefaultConfig):
    data_root = './dataset/yelp2016_data'
    # w2v_path = './dataset/yelp2016_data/train/npy/w2v.npy'

    # user_list_path = './dataset/yelp2016_data/train/npy/userReview2Index.npy'
    # item_list_path = './dataset/yelp2016_data/train/npy/itemReview2Index.npy'

    # user2itemid_path = './dataset/yelp2016_data/train/npy/user_item2id.npy'
    # item2userid_path = './dataset/yelp2016_data/train/npy/item_user2id.npy'
    user2review_path = './dataset/yelp2016_data/train/npy/user2review.npy'
    item2review_path = './dataset/yelp2016_data/train/npy/item2review.npy'
    vocab_size = 264045
    word_dim = 300

    r_max_len = 189

    u_max_r = 9
    i_max_r = 16
    u_avg_r = 857
    i_avg_r = 1405


    train_data_size = 1024505
    test_data_size = 121317

    user_num = 164179 + 2
    item_num = 100125 + 2

    batch_size = 32


class Instant_Video_data_Config(DefaultConfig):
    data_root = './dataset/Instant_Video_data'
    w2v_path = './dataset/Instant_Video_data/train/npy/w2v.npy'

    user_list_path = './dataset/Instant_Video_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Instant_Video_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Instant_Video_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Instant_Video_data/train/npy/item_user2id.npy'
    # user2review_path = './dataset/Tools_Improvement_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Tools_Improvement_data/train/npy/item2review.npy'

    item_user2Review_path = './dataset/Instant_Video_data/train/npy/user_item2Review.npy'
    ui2r_path = './dataset/Instant_Video_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Instant_Video_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Instant_Video_data/train/npy/i2ur.npy'

    vocab_size = 53009
    word_dim = 300

    r_max_len = 148

    u_max_r = 9
    i_max_r = 34

    train_data_size = 29700
    test_data_size = 3713

    user_num = 5130 + 2
    item_num = 1685 + 2

    batch_size = 64

class Tools_Improvement_data_Config(DefaultConfig):
    data_root = './dataset/Tools_Improvement_data'
    w2v_path = './dataset/Tools_Improvement_data/train/npy/w2v.npy'

    user_list_path = './dataset/Tools_Improvement_data/train/npy/userReview2Index.npy'
    item_list_path = './dataset/Tools_Improvement_data/train/npy/itemReview2Index.npy'

    user2itemid_path = './dataset/Tools_Improvement_data/train/npy/user_item2id.npy'
    item2userid_path = './dataset/Tools_Improvement_data/train/npy/item_user2id.npy'
    # user2review_path = './dataset/Tools_Improvement_data/train/npy/user2review.npy'
    # item2review_path = './dataset/Tools_Improvement_data/train/npy/item2review.npy'

    item_user2Review_path = './dataset/Tools_Improvement_data/train/npy/user_item2Review.npy'
    ui2r_path = './dataset/Tools_Improvement_data/train/npy/ui2r.npy'
    u2ir_path = './dataset/Tools_Improvement_data/train/npy/u2ir.npy'
    i2ur_path = './dataset/Tools_Improvement_data/train/npy/i2ur.npy'

    vocab_size = 74475
    word_dim = 300

    r_max_len = 194

    u_max_r = 11
    i_max_r = 20

    train_data_size = 107595
    test_data_size = 26891

    user_num = 16638 + 2
    item_num = 10217 + 2

    batch_size = 64

def save_user_list(opt):
    # user_reviews = opt.user_list
    user_item2id = opt.user2itemid_dict
    user_item2review = opt.user_item2review.item()
    
    user_li = []
    for uid in range(len(user_item2id)):
        u2iids = user_item2id[uid] 
        li = []
        for u2iid in u2iids:
            try:
                review_emb = user_item2review[uid][u2iid].view(-1).tolist()
            except:
                review_emb = [0] * 768
            li.append(review_emb)
            # ipdb.set_trace()
            # tensor_li = torch.stack(li)
        user_li.append(li)
    # ipdb.set_trace()
    np.save(opt.u2ir_path, user_li)

def save_item_list(opt):
    # user_reviews = opt.user_list
    item_user2id = opt.item2userid_dict
    user_item2review = opt.user_item2review.item()
    
    item_li = []
    for iid in range(len(item_user2id)):
        i2iids = item_user2id[iid] 
        li = []
        for i2iid in i2iids:
            try:
                review_emb = user_item2review[i2iid][iid].view(-1).tolist()
            except:
                review_emb = [0] * 768
            li.append(review_emb)
            # ipdb.set_trace()
            # tensor_li = torch.stack(li)
        item_li.append(li)
    # ipdb.set_trace()
    np.save(opt.i2ur_path, item_li)
# ====================================
# 数据集名称：Musical_Instruments_data
# trainData_size: 8214
# testData_size: 2047
# userNum: 1428
# itemNum: 899
# w2v词向量维度300
# w2v大小19054
# ====================================
# 数据集名称：Digital_Music_data
# trainData_size: 51764
# testData_size: 12942
# userNum: 5541
# itemNum: 3568
# u_r_len: 16
# i_r_len: 31
# w2v向量维度300
# w2v大小 107680
# ====================================
# 数据集名称：Tools_Improvement_data
# trainData_size: 107585
# testData_size: 26891
# userNum: 16636
# itemNum: 10216
# u_r_len: 11
# i_r_len: 20
# w2v向量维度300
# w2v大小 74475

