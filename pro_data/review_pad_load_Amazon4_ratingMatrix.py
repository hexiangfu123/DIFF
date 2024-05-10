# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import time
from sklearn.model_selection import train_test_split
import gensim
import torch
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import random
from collections import defaultdict
import ipdb
from transformers import BertModel, BertTokenizer
import os

p_review = 0.85
p_summary = 0.95
p_char = 0.8

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_count(data, id):
    # count,每个id参与评论的次数
    idList = data[[id, 'ratings']].groupby(id, as_index=False)
    idListCount = idList.size()
    return idListCount


def numerize(data):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data

def build_ui2bert_emb_dict(data):
    dict1 = {}
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda:0")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    # bert = BertModel.from_pretrained("bert-base-uncased")
    for x in data.index:
        # print(data.loc[x])
        user_id, item_id, rating, review = data.loc[x]
        input_text = clean_str(review).encode('ascii', 'ignore').decode('ascii')
        # tokens = tokenizer(input_text, padding = 'max_length', truncation=True, max_length=maxlen)
        tokenized_text = tokenizer.encode(input_text, max_length=512) #token初始化
        # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #获取词汇表索引
        tokens_tensor = torch.tensor([tokenized_text]).to(device) #将输入转化为torch的tensor
        with torch.no_grad(): #禁用梯度计算 因为只是前向传播获取隐藏层状态，所以不需要计算梯度 
            try:
                last_hidden_states = model(tokens_tensor)[0]
            except:
                ipdb.set_trace() 
        # ipdb.set_trace()
        reviews_emb = torch.max(last_hidden_states, dim = 1).values
        # 
        # emb = bert(tokens)
        # emb_cls = emb[1]
        if user_id not in dict1.keys():
            dict1[user_id] = {}
        dict1[user_id][item_id] = reviews_emb
        # ipdb.set_trace()
        
    return dict1
def build_ui2r_dict(data):
    dict1 = {}
    dict2 = {}
    for x in data.index:
        # print(data.loc[x])
        user_id, item_id, rating, review = data.loc[x]
        # ipdb.set_trace()
        input_text = clean_str(review).encode('ascii', 'ignore').decode('ascii')
        # ui.append([user_id, item_id, rating])
        if user_id in dict1.keys():
            dict1[user_id] += ' ' + input_text
        else:
            dict1[user_id] = input_text
        if item_id in dict2.keys():
            dict2[item_id] += ' ' + input_text
        else:
            dict2[item_id] = input_text
    cnt1, cnt2 = [], []
    for k in dict1.keys():
        # ipdb.set_trace()
        cnt1.append(len(dict1[k].split(' ')) + 1)
    list.sort(cnt1)
    for k in dict2.keys():
        # ipdb.set_trace()
        cnt2.append(len(dict2[k].split(' ')) + 1)
    list.sort(cnt2)
    
    # ipdb.set_trace()

    print("===============Start: User/Item Dict======================")
    print("Average user review length: {}".format(sum(cnt1)/len(cnt1)))
    print("User review length min:{} max:{}".format(cnt1[0], cnt1[-1]))
    print("Average item review length: {}".format(sum(cnt2)/len(cnt2)))
    print("Item review length min:{} max:{}".format(cnt2[0], cnt2[-1]))
    print("===============End: User/Item Dict======================")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    for k in dict1.keys():
        # ipdb.set_trace()
        dict1[k] = tokenizer(dict1[k], padding = 'max_length', truncation=True, max_length=512)
        # print(len(dict1[k]))
    for k in dict2.keys():
        dict2[k] = tokenizer(dict2[k], padding = 'max_length', truncation=True, max_length=512) 
    return dict1, dict2

def clean_str(string):
    # string = re.sub(r"\\","", string) #re.sub(),将文本中的"\"替换成空白
    # string = re.sub(r"\'","",string)
    # string = re.sub(r"\"", "",string)
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss "," ",string)
    return string.strip().lower()

def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        textTmp = ""
        for t in text:
            textTmp = textTmp + " " + t
        rawReviews.append(textTmp.strip())
    return rawReviews

def countSummary(xDict):
    maxSent = 0
    minSent = 3000
    for (i, text) in xDict.items():
        for sent in text:
            if sent != '':
                wordTokens = sent.split()
                if len(wordTokens) > maxSent:
                    maxSent = len(wordTokens)
                if len(wordTokens) < minSent:
                    minSent = len(wordTokens)
    return maxSent, minSent


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            # if len(wordTokens) > 1000:
            #     print sent
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    # #################以85%的覆盖率确定句子最大长度##########################
    # 将所有review的长度按照从小到大排序
    x = np.sort(SentLenList)
    # 统计有多少个评论
    xLen = len(x)
    # 以p覆盖率确定句子长度
    pSentLen = x[int(p_review * xLen) - 1]
    x = np.sort(ReviewLenList)
    # 统计有多少个评论
    xLen = len(x)
    # 以p覆盖率确定句子长度
    pReviewLen = x[int(p_review * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen


def build_char(rawReviews):
    all_chars = []
    word_length = []
    for r in rawReviews:
        tmp_chars = []
        words = r.split()
        for w in words:
            tmp_chars.extend(list(w))
            word_length.append(len(w))
        tmp_chars = list(set(tmp_chars))
        all_chars.extend(tmp_chars)
        all_chars = list(set(all_chars))

    word_length = sorted(word_length)
    size = len(word_length)
    pCharLen = word_length[int(size * p_char)]
    all_chars.append('<')
    all_chars.append('>')
    all_chars.append('$')

    char2id = {i: j for i, j in enumerate(all_chars)}
    id2char = {j: i for i, j in enumerate(all_chars)}
    return char2id, id2char, pCharLen
    # 以p 概率覆盖word


# #####################################1，数据数据索引化############################################


if __name__ == '__main__':
    assert(len(sys.argv) == 2)

    filename = sys.argv[1]
    # Data Sets  
        # Digital_Music_5.json
        # Gourmet_Food_5.json
        # Musical_Instruments_5.json  user:463 item:860
        # Office_Products_5.json
        # Video_Games_5.json
        # Tools_Improvement_5.json
        # Toys_and_Games_5.json
        # yelp2016.ss
    save_folder = '../dataset/' + filename[:-7] + "_data"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder + '/train/npy')
        os.makedirs(save_folder + '/test/npy')
    print("数据集名称：{}".format(save_folder))

    users_id = []
    items_id = []
    ratings = []
    reviews = []
    summarys = []

    # ------------------------------for yelp1314----------------------------------------------------
    # for line in file:
    #     value = line.split('\t\t')
    #     reviews.append(value[3])
    #     users_id.append(value[0])
    #     items_id.append(value[1])
    #     ratings.append(value[2])

    # --------------------------------------for yelp16----------------------------------------------
    if 'yelp' in filename:
        file1 = open('./rawData/' + filename + '/yelp_academic_dataset_business.json', errors='ignore')
        bussiness_id = []
        bussiness_all = []

        for line in file1:
            js = json.loads(line)
            cat = js["categories"]

            if cat != None and 'Restaurants' in cat:
                bussiness_id.append(js["business_id"])
            else:
                bussiness_all.append(js["business_id"])
        bussiness_all.extend(bussiness_id)
        bussiness_all = list(set(bussiness_all))
        bussiness_id = list(set(bussiness_id))
        file2 = open('./rawData/' + filename + '/yelp_academic_dataset_review.json', errors='ignore')
        ipdb.set_trace()
        for line in file2:
            js = json.loads(line)
            
            bu = js["business_id"] 
            reviews.append(js["text"])
            users_id.append(js["user_id"])
            items_id.append(js["business_id"])
            ratings.append(js["stars"])

    # ---------------------------------for amazon --------------------------------------------------
    elif '_5.json' in filename:
        file = open('./rawData/' + filename, errors='ignore')
        for line in file:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['asin']) == 'unknown':
                print("unkown item id")
                continue
            try:
                reviewText = js['reviewText']
                reviewerID = str(js['reviewerID'])
                asin = str(js['asin'])
                overall = str(js['overall'])
            except:
                continue
            reviews.append(reviewText)
            users_id.append(reviewerID)
            items_id.append(asin)
            ratings.append(overall)

    data = pd.DataFrame({'user_id': pd.Series(users_id),
                       'item_id': pd.Series(items_id),
                       'ratings': pd.Series(ratings),
                       'reviews': pd.Series(reviews)})[['user_id', 'item_id', 'ratings', 'reviews']]
    # ================释放内存============#
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    summarys = []
    # ====================================#
    userCount, itemCount = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_raw = userCount.shape[0]
    itemNum_raw = itemCount.shape[0]
    print("===============Start: rawData size======================")
    print("dataNum: {}".format(data.shape[0]))
    print("userNum: {}".format(userNum_raw))
    print("itemNum: {}".format(itemNum_raw))
    print("data densiy: {:.5f}".format(data.shape[0]/float(userNum_raw * itemNum_raw)))
    print("===============End: rawData size========================")
    uidList = userCount.user_id  # userID列表
    iidList = itemCount.item_id  # itemID列表
    user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
    data = numerize(data)
    ipdb.set_trace()
    ui2r = build_ui2bert_emb_dict(data) 
    np.save("./{}/train/npy/ui2r.npy".format(save_folder), ui2r)

    # ########################在构建字典库之前，先划分数据集###############################
    # data = data.sample(frac=1, random_state=1234)
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
    # 重新统计训练集中的用户数，商品数，查看是否有丢失的数据
    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uidList_train = userCount.index
    iidList_train = itemCount.index
    userNum = userCount.shape[0]
    itemNum = itemCount.shape[0]
    print("===============Start-no-process: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End-no-process: trainData size========================")

    uidMiss = []
    iidMiss = []
    if userNum != userNum_raw or itemNum != itemNum_raw:
        for uid in range(userNum_raw):
            if uid not in uidList_train:
                uidMiss.append(uid)
        for iid in range(itemNum_raw):
            if iid not in iidList_train:
                iidMiss.append(iid)

    if len(uidMiss):
        for uid in uidMiss:
            df_temp = data_test[data_test['user_id'] == uid]
            data_test = data_test[data_test['user_id'] != uid]
            data_train = pd.concat([data_train, df_temp])

    if len(iidMiss):
        for iid in iidMiss:
            df_temp = data_test[data_test['item_id'] == iid]
            data_test = data_test[data_test['item_id'] != iid]
            data_train = pd.concat([data_train, df_temp])

    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)
    # 重新统计训练集中的用户数，商品数，查看是否有丢失的数据
    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uidList_train = userCount.index
    iidList_train = itemCount.index
    userNum = userCount.shape[0]
    itemNum = itemCount.shape[0]
    print("===============Start-already-process: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End-already-process: trainData size========================")

    x_train = []
    y_train = []
    for i in data_train.values:
        uiList = []
        uid = i[0]
        iid = i[1]
        score = i[2]
        uiList.append(uid)
        uiList.append(iid)
        x_train.append(uiList)
        y_train.append(float(i[2]))

    x_val = []
    y_val = []
    for i in data_test.values:
        uiList = []
        uid = i[0]
        iid = i[1]
        uiList.append(uid)
        uiList.append(iid)
        x_val.append(uiList)
        y_val.append(float(i[2]))
    np.save("./{}/train/npy/Train.npy".format(save_folder), x_train)
    np.save("./{}/train/npy/Train_Score.npy".format(save_folder), y_train)
    np.save("./{}/test/npy/Test.npy".format(save_folder), x_val)
    np.save("./{}/test/npy/Test_Score.npy".format(save_folder), y_val)
    
    print("{} 测试集大小{}".format(now(), len(x_val)))
    print("{} 测试集评分大小{}".format(now(), len(y_val)))
    print("{} 训练集大小{}".format(now(), len(x_train)))

    # #####################################2，构建字典库，只针对训练数据############################################
    user_reviews_dict = {}
    item_reviews_dict = {}
    # 新增项
    review_summary_dict = {}

    user_iid_dict = {}
    item_uid_dict = {}
    user_len = defaultdict(int)
    item_len = defaultdict(int)

    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

        str_review_summary = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if len(str_review_summary.strip()) == 0:
            str_review_summary = "<unk>"

        if user_reviews_dict.__contains__(i[0]):
            user_reviews_dict[i[0]].append(str_review)
            review_summary_dict[i[0]].append(str_review_summary)
            user_iid_dict[i[0]].append(i[1])

        else:
            user_reviews_dict[i[0]] = [str_review]
            review_summary_dict[i[0]] = [str_review_summary]
            user_iid_dict[i[0]] = [i[1]]

        if item_reviews_dict.__contains__(i[1]):
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]

    # 构建字典库,User和Item的字典库是一样的
    rawReviews = bulid_vocbulary(review_summary_dict)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(rawReviews)
    word_index = tokenizer.word_index
    word_index['<unk>'] = 0
    print ("字典库大小{}".format(len(word_index)))
    id2char, char2id, pCharLen = build_char(rawReviews)
    np.save('id2char.npy', id2char)

    # 用户文本数统计
    print(now())
    print("字符个数：{}, 覆盖率{}, 单词字符数为: {}".format(len(char2id), p_char, pCharLen))
    u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen= countNum(user_reviews_dict)
    print ("用户最少有{}个评论,最多有{}个评论，平均有{}个评论, " \
         "句子最大长度{},句子的最短长度{}，" \
         "设定用户评论个数为{}： 设定句子最大长度为{}".format(u_minNum,u_maxNum,u_averageNum,u_maxSent, u_minSent, u_pReviewLen, u_pSentLen))
    # 商品文本数统计
    i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen= countNum(item_reviews_dict)
    print ("商品最少有{}个评论,最多有{}个评论，平均有{}个评论," \
         "句子最大长度{},句子的最短长度{}," \
         ",设定商品评论数目{}, 设定句子最大长度为{}".format(i_minNum,i_maxNum,i_averageNum,u_maxSent,i_minSent,i_pReviewLen, i_pSentLen))
    print ("最终设定句子最大长度为(取最大值)：{}".format(max(u_pSentLen,i_pSentLen)))

    # ########################################################################################################
    
    maxSentLen = max(u_pSentLen, i_pSentLen)
    minSentlen = 1
    userReview2Index = []
    userReviewChar2Index = []
    user_iid_list = []
    ui_review = {}
    max1 = 0
    for i in user_reviews_dict.keys():
        if max1 < i:
            max1 = i
    for i in range(max1 + 1):
        count_user = 0
        dataList = []
        dataList_summary = []
        a_count = 0

        try:
            textList = user_reviews_dict[i]
            u_iids = user_iid_dict[i]
        except:
            textList = []
            u_iids = []
        
        u_reviewList = []  # 待添加
        u_reviewLen = []   # 待添加
        u_reviewChar = []
        ui_review[i] = {}

        def padding_chars(u_textCharList, num1, num2):
            tmp = [[char2id['$']] * pCharLen] * num1
            if len(u_textCharList) >= num2:
                return u_textCharList[:num2]
            else:
                return u_textCharList + [tmp for _ in range(num2 - len(u_textCharList))]

        def padding_text(textList, num):
            new_textList = []
            if textList == []:
                return new_textList
            if len(textList) >= num:
                new_textList = textList[:num]
            else:
                padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
                new_textList = textList + padding
            return new_textList

        def padding_ids(iids, num, pad_id):
            if len(iids) >= num:
                new_iids = iids[:num]
            else:
                new_iids = iids + [pad_id] * (num - len(iids))
            return new_iids

        user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum+1))
        
        for j in range(len(textList)):
            text = textList[j]
            iid = u_iids[j]
            text2index = []
            textChar2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) >= minSentlen:
                k = 0
                if len(wordTokens) > maxSentLen:
                    u_reviewLen.append(maxSentLen)
                else:
                    u_reviewLen.append(len(wordTokens))
                for _, word in enumerate(wordTokens):
                    if k < maxSentLen:
                        text2index.append(word_index[word])
                        k = k + 1
                    else:
                        break
                    wordChar2Index = [char2id[c] for c in list(word)]
                    # padding word char
                    if len(wordChar2Index) < pCharLen:
                        wordChar2Index = wordChar2Index + [char2id['$']] * (pCharLen - len(wordChar2Index))
                    else:
                        wordChar2Index = wordChar2Index[:pCharLen]
                    textChar2index.append(wordChar2Index)
            else:
                count_user += 1
                u_reviewLen.append(1)
            if len(text2index) < maxSentLen:
                textChar2index = textChar2index + [[char2id['$']] * pCharLen] * (maxSentLen - len(text2index))
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            ui_review[i][iid] = text2index
            u_reviewList.append(text2index)
            u_reviewChar.append(textChar2index)
        if count_user >= 1:
            print("第{}个用户共有{}个商品评论，经处理后有{}个为空".format(i, len(textList), count_user))

        userReviewChar2Index.append(padding_chars(u_reviewChar, maxSentLen, u_pReviewLen))
        if padding_text(u_reviewList, u_pReviewLen) == []:
            li = [0] * maxSentLen
            li = [li] * u_pReviewLen
        else:
            li = padding_text(u_reviewList, u_pReviewLen)
        userReview2Index.append(li)

    itemReview2Index = []
    itemReviewChar2Index = []
    item_uid_list = []

    max1 = 0
    for i in item_reviews_dict.keys():
        if max1 < i:
            max1 = i
    for i in range(max1 + 1):
        count_item = 0
        dataList = []
        dataList_summary = []
        try:
            textList = item_reviews_dict[i]
            i_uids = item_uid_dict[i]
        except:
            textList = []
            i_uids = []
        i_reviewList = []  # 待添加
        i_reviewChar = []
        i_reviewLen = []  # 待添加
        item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum+1))

        for text in textList:
            text2index = []
            textChar2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) >= minSentlen:
                k = 0
                if len(wordTokens) > maxSentLen:
                    i_reviewLen.append(maxSentLen)
                else:
                    i_reviewLen.append(len(wordTokens))
                for _, word in enumerate(wordTokens):
                    if k < maxSentLen:
                        text2index.append(word_index[word])
                        k = k + 1
                    else:
                        break
                    wordChar2Index = [char2id[c] for c in list(word)]
                    if len(wordChar2Index) < pCharLen:
                        wordChar2Index = wordChar2Index + [char2id['$']] * (pCharLen - len(wordChar2Index))
                    else:
                        wordChar2Index = wordChar2Index[:pCharLen]
                    textChar2index.append(wordChar2Index)
            else:
                count_item += 1
                i_reviewLen.append(1)
            if len(text2index) < maxSentLen:
                textChar2index = textChar2index + [[char2id['$']] * pCharLen] * (maxSentLen - len(text2index))
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            i_reviewList.append(text2index)
            i_reviewChar.append(textChar2index)
        if count_item >= 1:
            print("第{}个商品共有{}个用户评论,经处理后{}个为空".format(i, len(textList), count_item))

        itemReviewChar2Index.append(padding_chars(i_reviewChar, maxSentLen, i_pReviewLen))
        if padding_text(i_reviewList, i_pReviewLen) == []:
            li = [0] * maxSentLen
            li = [li] * i_pReviewLen
        else:
            li = padding_text(i_reviewList, i_pReviewLen)
        itemReview2Index.append(li)
    
    print("{} start writing npy...".format(now()))
    np.save("./{}/train/npy/userReview2Index.npy".format(save_folder), userReview2Index)
    np.save("./{}/train/npy/user_item2id.npy".format(save_folder), user_iid_list)
    np.save("./{}/train/npy/user_item2Review.npy".format(save_folder), ui_review)
    np.save("./{}/train/npy/itemReview2Index.npy".format(save_folder), itemReview2Index)
    np.save("./{}/train/npy/item_user2id.npy".format(save_folder), item_uid_list)
    print("{} write finised".format(now()))
    ########################################################################################################
