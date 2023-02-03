import argparse
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys

from collections import defaultdict
from dateutil import parser
from msft_baseline_features import address, CPred, CProp, HistIndiv, HistPair, Meta, MetaAdded, Temporal, User
from msft_baseline_features import BOW, pre_HistIndiv, pre_HistPair
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from utils import dict_df, dict_float, dict_int, dict_list, dict_time
from utils import load_pickles_or_run_func, time_to_hours
from utils import error_cal, save_error
from utils import draw_curve_seaborn, draw_hist_seaborn
def LR(X,Y):
    lr = LogisticRegression( max_iter = 500)
    lr.fit(X,Y)
    return lr

def ADA(X,Y):
    ada = AdaBoostClassifier()
    ada.fit(X,Y)
    return ada

def y_classification(y):
    y_class = y // 10 + int(y % 10 > 0)
    maximum = 24*7 // 50 + int(24*7 % 50 > 0)
    y_class = min(maximum, y_class)
    return int(y_class)

def enron_add_department_and_stage(email_item):
    if 'enron' in email_item['From']:
        department = 0
    else:
        department = 1
    try:
        stage = int(dic_stage[email_item['From']])
    except:
        stage = 6
    return department, stage

def get_enron_data_points_down_sampling(path, down_sampling_rate = 0.3):
    # get name -> email dic, stage initialization
    res_HistIndiv = load_pickles_or_run_func(path + "HistIndivALL.pkl", pre_HistIndiv, csv_file)
    res_HistPair = load_pickles_or_run_func(path + "HistPairALL.pkl", pre_HistPair, csv_file)
    
    # establish datasets
    data_points = [[], []]
    reply_time_list = defaultdict(dict_list) # reply_time_list[u_i][u_j] u_i is sender, u_j is recipient
    
    for i in range(len(train_csv_file)):
        res = []
        email_item = train_csv_file.iloc[i].to_dict()
        # get department and stage
        department, stage = enron_add_department_and_stage(email_item)
        email_item['Department'] = department
        email_item['Stage'] = stage

        cur_time = parser.parse(email_item['Date'])
        # res is the feature
        res = []
        for func in funcs[data_option]:
            res += func(email_item)
        res += HistIndiv(email_item, res_HistIndiv)
        res += HistPair(email_item, res_HistPair)
        
        IsInternalExternal, NumOfRecipients = address(email_item)
        
        # flatten res
        new_res = []
        for r in res:
            if type(r) is list:
                new_res += r
            else:
                new_res.append(r)
        res = new_res
        # extract from and to
        from_email = email_item['From']
        # to_email may be nan
        try:
            to_emails = email_item['To'].split()
        except:
            to_emails = []
        # add to reply_time_list
        # update to data points
        for IsInternalExternal_i, NumOfRecipients_i, to_email in zip(IsInternalExternal, NumOfRecipients, to_emails):
            # update reply_time
            reply_time_list[from_email][to_email] = res + [IsInternalExternal_i, NumOfRecipients_i, cur_time]
            # check whether add data_points
            if from_email in reply_time_list[to_email]: # from_email replies to_email now
                x = reply_time_list[to_email][from_email][:-1]
                y = time_to_hours(cur_time - reply_time_list[to_email][from_email][-1])
                data_points[0].append(x)
                data_points[1].append(y)
                del(reply_time_list[to_email][from_email])
    return data_points

def get_enron_data_points(path):
    # get name -> email dic, stage initialization
    res_HistIndiv = load_pickles_or_run_func(path + "HistIndivALL.pkl", pre_HistIndiv, csv_file)
    res_HistPair = load_pickles_or_run_func(path + "HistPairALL.pkl", pre_HistPair, csv_file)
    
    # establish datasets
    data_points = [[], []]
    reply_time_list = defaultdict(dict_list) # reply_time_list[u_i][u_j] u_i is sender, u_j is recipient
    
    for i in range(len(train_csv_file)):
        res = []
        email_item = train_csv_file.iloc[i].to_dict()
        # get department and stage
        department, stage = enron_add_department_and_stage(email_item)
        email_item['Department'] = department
        email_item['Stage'] = stage

        cur_time = parser.parse(email_item['Date'])
        # res is the feature
        res = []
        for func in funcs[data_option]:
            res += func(email_item)
        res += HistIndiv(email_item, res_HistIndiv)
        res += HistPair(email_item, res_HistPair)
        
        IsInternalExternal, NumOfRecipients = address(email_item)
        
        # flatten res
        new_res = []
        for r in res:
            if type(r) is list:
                new_res += r
            else:
                new_res.append(r)
        res = new_res
        # extract from and to
        from_email = email_item['From']
        # to_email may be nan
        try:
            to_emails = email_item['To'].split()
        except:
            to_emails = []
        # add to reply_time_list
        # update to data points
        for IsInternalExternal_i, NumOfRecipients_i, to_email in zip(IsInternalExternal, NumOfRecipients, to_emails):
            # update reply_time
            reply_time_list[from_email][to_email] = res + [IsInternalExternal_i, NumOfRecipients_i, cur_time]
            # check whether add data_points
            if from_email in reply_time_list[to_email]: # from_email replies to_email now
                x = reply_time_list[to_email][from_email][:-1]
                y = time_to_hours(cur_time - reply_time_list[to_email][from_email][-1])
                data_points[0].append(x)
                data_points[1].append(y)
                del(reply_time_list[to_email][from_email])
    return data_points

def get_enron_test_chat_file(test_csv_file):
    chat = defaultdict(list)
    num = 0
    for i in range(len(test_csv_file)):
        email_item= test_csv_file.iloc[i].to_dict()
        from_email = email_item['From']
        # to_email may be nan
        try:
            to_emails = email_item['To'].split()
        except:
            to_emails = []
        for to_email in to_emails:
            num += 2
            chat[from_email].append((from_email, to_email,email_item))
            chat[to_email].append((from_email,to_email,email_item))
    return chat

def get_feature(email_item, from_email, to_email, res_HistIndiv, res_HistPair):
    res = []
    for func in funcs[data_option]:
        res += func(email_item)
    res += HistIndiv(email_item, res_HistIndiv)
    res += HistPair(email_item, res_HistPair)
    
    _, NumOfRecipients = address(email_item)
    IsInternalExternal = 1 if 'enron' in from_email and 'enron' in to_email else 0
    
    # flatten res
    new_res = []
    for r in res:
        if type(r) is list:
            new_res += r
        else:
            new_res.append(r)
    res = new_res
    res += [IsInternalExternal, NumOfRecipients[0]]
    return res 

def check_expire_time(cur_time_date, send_time, reply_time, expire_window = 3):
    for key in send_time:
        for i in range(len(send_time[key])):
            if (cur_time_date - send_time[key][i][-1].date()).days < expire_window:
                break
        i = i-1
        # delete the timeline before expire_window
        send_time[key] = send_time[key][i:]
        if len(send_time[key]) == 0:
            del(send_time[key])
    key_to_del = []
    for key in reply_time.keys():
        for i in range(len(reply_time[key])):
            if (cur_time_date - reply_time[key][i][-1].date()).days < expire_window:
                break
        i = i-1
        # delete the timeline before expire_window
        reply_time[key] = reply_time[key][i:]
        if len(reply_time[key]) == 0:
            key_to_del.append(key)
    # delete keys don't exist
    for key in key_to_del:
        del(reply_time[key])
    return send_time, reply_time

def prediction(X, model):
    if X == {}:
        return {}
    dic = {}
    # simplify
    X_together = []
    e_j_together = []
    for e_j in X:
        for item in X[e_j]:
            X_together.append(item[0])
            e_j_together.append(e_j)
    res = model.predict(X_together)
    for e_j, r in zip(e_j_together,res):
        if e_j not in dic:
            dic[e_j] = r
        dic[e_j] = min(dic[e_j], r)
    # for e_j in X:
    #     for item in X[e_j]:
    #         res = int(model.predict([item[0]]))
    #         if e_j not in dic:
    #             dic[e_j] = res
    #         dic[e_j] = min(dic[e_j], res)
    res = sorted([(val, key) for key, val in dic.items()])
    dic = {}
    for i in range(len(res)):
        dic[res[i][1]] = i+1
    return dic



def test_enron(path, model):
    # set expire time. same like what we did before
    expire_time = 3
    dic_expire = defaultdict(dict_list)
    
    # begin data load
    res_HistIndiv = load_pickles_or_run_func(path+"enron_msft_HistIndivALL.pkl", pre_HistIndiv, csv_file)
    res_HistPair = load_pickles_or_run_func(path+"enron_msft_HistPairALL.pkl", pre_HistPair, csv_file)
    if args.year == 0:
        if start_i == 0:
            file_name = "enron_msft_test_ALL"
        else:
            file_name = "enron_msft_test_PART"
    else:
        file_name = "enron_msft_test_year" + str(args.year)
    if args.down_sampling == 0:
        test_enron_chat_file = load_pickles_or_run_func(path + file_name + '.pkl', get_enron_test_chat_file, test_csv_file)
    else:
        # down_sampling
        file_name += "_ds" + str(args.down_sampling)
        # datapoints = load_pickles_or_run_func(path + file_name + '.pkl', get_enron_data_points_down_sampling, path)
        test_enron_chat_file = load_pickles_or_run_func(path + file_name + '.pkl', get_enron_test_chat_file, test_csv_file)
    # total length 322337
    # initilization
    send_time = {}
    tmp_send_time = {}
    reply_time = {}

    err = []
    err_by_date = defaultdict(list)
    ndcg_by_date = defaultdict(list)
    ndcgs = []
    stat_pred = []
    stat_true = []
    num = 0
    for e_i in list(test_enron_chat_file.keys()):
        prev_time = parser.parse(test_enron_chat_file[e_i][0][2]['Date'])
        # use cur_day_receive and prev_day_send
        replied_today = []
        sent_today = []
        prev_e_i = e_i
        chat_e_i = defaultdict(list)
        for i, item in enumerate(test_enron_chat_file[e_i]):
            num += 1
            if num % 1000 == 0:
                print(num)
            # extract items we need
            from_email, to_email, email_item = item[0], item[1], item[2]
            # link from_email/to_email with e_i/e_j
            e_j = to_email if e_i == from_email else from_email
            cur_time = parser.parse(email_item['Date'])
            chat_e_i[e_j].append(email_item)
            chat_history = chat_e_i[e_j]
            if cur_time.date() != prev_time.date():      
                # we adopt reply_time as X, tmp_send_time as Y
                X = reply_time
                res_pred = prediction(reply_time, model)
                tmp_err, stat_pred_acc_per, stat_true_per, tmp_ndcgs = error_cal(res_pred, tmp_send_time, max_d = args.max_d)
                err_by_date[prev_time.date()].append(tmp_err)
                ndcg_by_date[prev_time.date()].append(tmp_ndcgs)
                err.append(tmp_err)
                ndcgs.append(tmp_ndcgs)
                stat_pred.append(stat_pred_acc_per)
                stat_true.append(stat_true_per)
                # following issues: update send time, delete expire time
                prev_time = cur_time
                # update send time
                for e_j, res in tmp_send_time.items():
                    # update tmp_send_time to send time
                    if e_j not in send_time:
                        send_time[e_j] = []
                    send_time[e_j] += res
                    # delete waiting to reply
                    if e_j in reply_time:
                        reply_time[e_j] = reply_time[e_j][len(res):]
                # delete expire time
                send_time, reply_time = check_expire_time(cur_time.date(), send_time, reply_time, expire_window = 3)
                tmp_send_time = {}

            # begin model
            
            # caculate res
            
            # update the stage and deparment
            department, stage = enron_add_department_and_stage(email_item)
            email_item['Department'] = department
            email_item['Stage'] = stage
            res = get_feature(email_item, from_email, to_email, res_HistIndiv, res_HistPair)
            
            # combine res with time
            res = [res, cur_time]
            # get address group
            # e_i send emails to others
            if e_i == from_email:
                if e_j not in tmp_send_time:
                    tmp_send_time[e_j] = []
                tmp_send_time[e_j].append(res)
            # e_i receive emails from others
            else:
                if e_j not in reply_time:
                    reply_time[e_j] = []
                
                reply_time[e_j].append(res)
                # since we receive the reply from others, update self.send_time
                if e_j in send_time:
                    reply_time[e_j] = reply_time[e_j][1:]
                    if len(reply_time[e_j]) == 0:
                        del(reply_time[e_j])
    return err, err_by_date, stat_pred, stat_true, ndcgs, ndcg_by_date

def down_sampling_csv(file):
    selected_index = []
    for i in range(len(file)):
        if args.down_sampling != 0:
            p = random.uniform(0, 1)
            if p < args.down_sampling:
                continue
            selected_index.append(i)
    return file.iloc[selected_index]

def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("--max_d", "-md", type=int, default = 10, help="Int parameter")
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0)
    parser.add_argument("-part_or_all", "-poa", type = str, default= 'part')
    parser.add_argument("-year", "-year", type = int, default= 0)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    random.seed(1)
    args = parse()
    idx_figure = args.part_or_all + "max_d" + str(args.max_d)
    if args.part_or_all == 'part':
        start_i = 400000
    else:
        start_i = 0
    print(start_i)
    if args.year != 0 and args.year  in [1999, 2000, 2001, 2002]:
        year_dic = {1999: [1138, 12281], 2000:[12282, 208382], 2001:[208383, 481414], 2002:[481415, 517319]}
        start_i, end_i = year_dic[args.year]
        idx_figure = 'year' + str(args.year) + "max_d" + str(args.max_d)
    pre_funcs = ['BOW','HistIndiv','pre_HistPair']
    # address is different need to be specified to every recipient
    funcs = {'enron':[ CPred, CProp, Temporal, User],  
    'avocado':[ CPred, CProp, Meta, MetaAdded, Temporal, User]}
    path = '../data/'
    dataset = {'enron':'sorted_emails1.csv'}
    # read original CSV
    data_option = 'enron'
    
    if args.down_sampling != 0:
        idx_figure += "down_sampling" + str(args.down_sampling)
    print(idx_figure)
    # load HistIndiv
    if data_option == 'enron':
        # load organ for dic
        org = pd.read_csv(path + 'organazition2.csv')
        dic_stage = {}
        for i in range(len(org)):
            if math.isnan(org.iloc[i]['Stage']):
                dic_stage[org.iloc[i]['Email'] ] = 6
            else:
                dic_stage[org.iloc[i]['Email'] ] = org.iloc[i]['Stage'] 
    
        csv_file = pd.read_csv(path + dataset[data_option])
        csv_file = csv_file.iloc[start_i:]
        iloc_train = int(0.8*len(csv_file))
        train_csv_file = csv_file.iloc[:iloc_train]
        test_csv_file = csv_file.iloc[iloc_train:]
        print(len(train_csv_file),len(test_csv_file))
        if args.year == 0:
            if start_i == 0:
                file_name = "enron_msft_train_ALL"
            else:
                file_name = "enron_msft_train_PART"
        else:
            file_name = "enron_msft_train_year" + str(args.year)
        if args.down_sampling == 0:
            print(file_name)
            datapoints = load_pickles_or_run_func(path + file_name + '.pkl', get_enron_data_points, path)
        else:
            # down_sampling
            train_csv_file = down_sampling_csv(train_csv_file)
            test_csv_file = down_sampling_csv(test_csv_file)
            file_name += "_ds" + str(args.down_sampling)
            print(file_name)
            datapoints = load_pickles_or_run_func(path + file_name + '.pkl', get_enron_data_points_down_sampling, path)
        
        X, Y = datapoints[0], datapoints[1] # X is all the features, Y is the reply time(by hours)
        for i in range(len(X)):
            X[i] = [float(x) for x in X[i]]
        Y_class = [y_classification(y) for y in Y]
        # train lr
        print("Start LR")
        lr = LR(X, Y_class)
        print("finish training")
        # test
        err, err_by_date, stat_pred, stat_true, ndcgs, ndcg_by_date = test_enron(path, lr)
        error_by_date_sum = []
        keys = sorted(err_by_date.keys())
        tmp_error_sum = []
        for key in keys:
            tmp_error_sum.append(sum(err_by_date[key])/len(err_by_date[key]))
        error_by_date_sum.append(tmp_error_sum)
        pre_s = 'msft_lr'+ str(idx_figure)
        save_error(path, pre_s, err, err_by_date, error_by_date_sum, stat_pred, stat_true, ndcgs, ndcg_by_date)
        draw_hist_seaborn([err], path, pre_s, idx_figure, s = ['msft_lr'])
        draw_curve_seaborn(error_by_date_sum, path, pre_s, idx_figure, s = ['msft_lr'])
        # train ada
        print("Start ADA")
        ada = ADA(X, Y_class)
        print("finish training")
        # test
        err, err_by_date, stat_pred, stat_true, ndcgs, ndcg_by_date = test_enron(path, ada)
        error_by_date_sum = []
        keys = sorted(err_by_date.keys())
        tmp_error_sum = []
        for key in keys:
            tmp_error_sum.append(sum(err_by_date[key])/len(err_by_date[key]))
        error_by_date_sum.append(tmp_error_sum)
        pre_s = 'msft_ada' + str(idx_figure)
        save_error(path, pre_s, err, err_by_date, error_by_date_sum, stat_pred, stat_true, ndcgs, ndcg_by_date)
        draw_hist_seaborn([err], path, pre_s, idx_figure, s = ['msft_ada'])
        draw_curve_seaborn(error_by_date_sum, path, pre_s, idx_figure, s = ['msft_ada'])