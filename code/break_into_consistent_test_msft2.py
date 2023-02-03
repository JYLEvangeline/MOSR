# break users into stationary vs non-stationary groups. For every time step, how many people is larger than 2. 
# 2001.10.1
import argparse
import math
import json
import os
import pandas as pd
import re
import pickle

from collections import defaultdict
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import *
from nltk.corpus import stopwords



from draw_year_heat_map_distribution import dic_set, get_name, get_enron_name, get_stage, get_corr, get_sqrt_diff, online_two_path, remove_stop_words, sort2
from train_test_differently_msft import check_expire_time, enron_add_department_and_stage, get_feature, prediction
from msft_baseline_features import pre_HistIndiv, pre_HistPair
from utils import load_pickles_or_run_func, error_cal
def get_date_iloc(csv_file, start_date, start_i = 0):
    """ get the start iloc of curren date

    Args:
        csv_file (DataFrame): the csv file of all records
        start_date (date.datetime): the start date

    Returns:
        int: the iloc of related date
    """
    year_dic = {'1979': 522, '1986': 524, '1997': 961, '1998': 1138, '1999': 12282, '2000': 208383, '2001': 481415, '2002': 517320, '2004': 517390, '2005': 517391, '2007': 517392, '2012': 517394, '2020': 517396, '2024': 517397, '2043': 517398, '2044': 517401}
    if start_i == 0:
        start_i = year_dic[str(start_date.year-1)]
    for i in range(start_i, len(csv_file)):
        cur_time = parser.parse(csv_file.iloc[i]['Date'])
        if cur_time.date() < start_date.date():
            continue
        if cur_time.date() == start_date.date():
            return i 
        if cur_time.date() > start_date.date():
            return i - 1

def test_enron(test_enron_chat_file, model, path = '../data/',csv_file = 0):
    
    # begin data load
    res_HistIndiv = load_pickles_or_run_func(path+"enron_msft_HistIndivALL.pkl", pre_HistIndiv, csv_file)
    res_HistPair = load_pickles_or_run_func(path+"enron_msft_HistPairALL.pkl", pre_HistPair, csv_file)
    
    # initilization
    send_time = {}
    tmp_send_time = {}
    reply_time = {}
    err = []
    ndcgs = []
    loss_by_user = {}
    ndcg_by_user = {}

    num = 0
    for e_i in list(test_enron_chat_file.keys()):
        loss_by_user[e_i] = []
        ndcg_by_user[e_i] = []
        prev_time = parser.parse(test_enron_chat_file[e_i][0][2]['Date'])
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
            if cur_time.date() != prev_time.date():      
                # we adopt reply_time as X, tmp_send_time as Y
                X = reply_time
                res_pred = prediction(reply_time, model)
                tmp_err, stat_pred_acc_per, stat_true_per, tmp_ndcgs = error_cal(res_pred, tmp_send_time)
                err.append(tmp_err)
                ndcgs.append(tmp_ndcgs)
                loss_by_user[e_i].append(tmp_err)
                ndcg_by_user[e_i].append(tmp_ndcgs)
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
    # return err, err_by_date, stat_pred, stat_true, ndcgs
    for key in loss_by_user.keys():
        try:
            loss_by_user[key] = (sum(loss_by_user[key])/len(loss_by_user[key]), len(loss_by_user[key]))
        except:
            loss_by_user[key] = ('NA',0)
    for key in ndcg_by_user.keys():
        try:
            ndcg_by_user[key] = (sum(ndcg_by_user[key])/len(ndcg_by_user[key]), len(ndcg_by_user[key]))
        except:
            ndcg_by_user[key] = ('NA',0)
    return err, ndcgs, loss_by_user, ndcg_by_user


def build_test_set(start_date, delta, dic, csv_file):
    chat = defaultdict(list)
    end_date = start_date + delta
    num = 0
    if start_date not in dic:
        iloc = get_date_iloc(csv_file, start_date)
        dic[start_date] = iloc
    start_iloc = dic[start_date]
    if end_date not in dic:
        iloc = get_date_iloc(csv_file, end_date, start_i = start_iloc)
        dic[end_date] = iloc
    end_iloc = dic[end_date]
    for i in range(start_iloc, end_iloc + 1):
        email_item = csv_file.iloc[i].to_dict()
        cur_time = parser.parse(email_item['Date'])
        if cur_time.date() < start_date.date():
            continue
        if cur_time.date() > end_date.date():
            break
        email_item= csv_file.iloc[i].to_dict()
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
    return chat, dic
     

def get_enron_test_chat_file(test_csv_file):
    chat = defaultdict(list)
    num = 0
    for i in range(len(test_csv_file)):
        email_item = test_csv_file.iloc[i].to_dict()
        cur_time = parser.parse(email_item['Date'])
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

def start_date_to_str(start_date):
    date_in_list = map(str, [start_date.year, start_date.month, start_date.day])
    return '.'.join(date_in_list)

def construct_matrix_dic(dic1, dic2, keys):    
    diff_dic = {}
    i = 0
    for key in keys:
        i +=1 
        if i%100000==0:
            print(i)
        corr1 = get_corr(dic1[key])
        corr2 = get_corr(dic2[key])
        sqrt_diff = get_sqrt_diff(corr1, corr2)
        diff_dic[key] = sqrt_diff
    return diff_dic

def seperate_candidates(json_diff_dic, seperate_val = 2):
    """ 
    Seperate the candidates based on their corr
    Args:
        json_diff_dic (dict): The corr for each candidate between two time periods
    """
    non_stationary = []
    stationary = []
    for key, val in json_diff_dic.items():
        if val <= seperate_val:
            stationary.append(key)
        else:
            non_stationary.append(key)
    return non_stationary, stationary

def seperate_to_stationary_and_non_stationary_dict(non_stationary, stationary, dic):
    dic_non_stationary = {}
    dic_stationary = {}
    non_stationary = set(non_stationary)
    stationary = set(stationary)
    for key, val in dic.items():
        if key in non_stationary:
            dic_non_stationary[key] = val
        else:
            dic_stationary[key] = val
    # json_non_stationary = json.dumps(dic_non_stationary)
    # json_stationary = json.dumps(dic_stationary)
    return dic_non_stationary, dic_stationary

def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("--max_d", "-md", type=int, default = 10, help="Int parameter")
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0)
    parser.add_argument("-part_or_all", "-poa", type = str, default= 'all')
    parser.add_argument("-start_date", "-sd", type = str, default = "2001.10.1")
    parser.add_argument("-train_length","-trainl", type = int, default = 120) # days
    parser.add_argument("-seperate_val","-sv", type = int, default = 1) 
    args = parser.parse_args()

    return args

def main():
    args = parse()
    delta = timedelta(days = args.train_length)
    start_date = args.start_date.split(".")
    delta = timedelta(days = args.train_length)
    start_date = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]))
    dic = defaultdict(int)
    path = '../data/'
    dataset = ['soc-sign-bitcoinotc.csv',\
        'soc-redditHyperlinks-body.tsv',\
            'soc-redditHyperlinks-title.tsv',\
                'email-Eu-core-temporal.txt',\
                    'CollegeMsg.txt',\
                    'enron_cmu.csv',\
                    'sorted_enron.csv',\
                        'emails.csv',\
                            'sorted_emails1.csv']
    # read csv file and build dic stage
    dic = {} # dic to record iloc
    csv_file = pd.read_csv(path + dataset[8])
    org = pd.read_csv(path + 'organazition2.csv')
    dic_stage = {}
    for i in range(len(org)):
        dic_stage[org.iloc[i]['Email']] = org.iloc[i]['Stage'] 
    start_date -= relativedelta(months=+5)
    # build json file
    all_json = []
    json_names = []
    
    for i in range(1,7):
        start_date += relativedelta(months=+1)
        json_names.append(start_date_to_str(start_date))
        file_path = path + 'break_into_consistent/msft2/' + start_date_to_str(start_date) + "delta" + str(args.train_length)
        if os.path.exists(file_path) == False:
            # create json file for current time period
            test_file, dic = build_test_set(start_date, delta, dic, csv_file)
            json_people = json.dumps(test_file)
            open(file_path, "w").write(json_people)
        all_json.append(file_path)
        if i == 1:
            start_date += relativedelta(months=+3)

    # json file at the last time period in train
    json_in_train, json_name_in_train = all_json[0], json_names[0]
    # json files in test
    all_json, json_names = all_json[1:], json_names[1:]
    
    print(json_names)
    
    model_lr = pickle.load(open("../data/model/differently/msft/msft_ada1999.1.11004.model","rb"))
    
    print("Start LR")
    for json_path_in_test, json_name_in_test in zip(all_json, json_names):
        print(json_name_in_test)
        # calculate results
        # read lr and ada model
        res_path = path + 'break_into_consistent/msft_lr_result/' + json_name_in_train + 'vs' + json_name_in_test + "seperate_val" + str(args.seperate_val)
        if os.path.exists(res_path) == False:
            json_file = json.load(open(json_path_in_test, "r"))
            res = test_enron(json_file, model_lr)
            del(json_file)
            with open(res_path, "wb") as fp:   
                pickle.dump(res, fp)

    del(model_lr)    

    model_ada = pickle.load(open("../data/model/differently/msft/msft_lr1999.1.11004.model","rb"))

    print("Start Ada")
    for json_path_in_test, json_name_in_test in zip(all_json, json_names):
        print(json_name_in_test)
        # calculate results
        # read lr and ada model
        res_path = path + 'break_into_consistent/msft_ada_result/' + json_name_in_train + 'vs' + json_name_in_test + "seperate_val" + str(args.seperate_val)
        if os.path.exists(res_path) == False:
            json_file = json.load(open(json_path_in_test, "r"))
            res = test_enron(json_file, model_ada)
            with open(res_path, "wb") as fp:   
                pickle.dump(res, fp)
    
        
    

if __name__ == "__main__":
    main()