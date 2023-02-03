from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import math

from pandas import DataFrame

"""
pre-defined defaultdict type
"""
def float_7():
    return 24.0*7

def dict_df():
    return defaultdict(DataFrame)

def dict_float():
    return defaultdict(float_7)

def dict_int():
    return defaultdict(int)

def dict_list():
    return defaultdict(list)

def dict_time():
    return defaultdict(datetime)

"""
get mean min and max
"""
def get_mean_min_max(l):
    if len(l) > 0:
        mean, min, max = 0, l[0], l[0]
        for item in l:
            mean += item
            if item < min:
                min = item
            if item > max:
                max = item
        return [mean, min, max]
    else:
        return [0, 0, 0]

"""
time to hours
"""
def time_to_hours(time):
    days, seconds = time.days, time.seconds
    return days*24 + seconds/3600

"""
load pickle with functions

"""
def load_pickles_or_run_func(pkl_path, func, csv_file):
    if os.path.exists(pkl_path):
        pkl_file = open(pkl_path, 'rb')
        res = pickle.load(pkl_file)
        pkl_file.close()
    else:
        res = func(csv_file)
        pkl_file = open(pkl_path, 'wb')
        pickle.dump(res, pkl_file)
    return res

"""
clean data utils
"""
def get_mean_med_time_list(time_list):
    l = sorted(time_list)
    # HistReplyTimeMeanGlobalUI
    sum_of_time = timedelta(hours = 0)
    for time in l:
        sum_of_time += time
    avg_of_time = sum_of_time / len(l)
    
    # HistReplyTimeMedianGlobalUI
    l = sorted(l)
    n = len(l) // 2
    if len(l) % 2 == 0:
        med_of_time = (l[n] + l[n-1])/2
    else:
        med_of_time = l[n]
    return avg_of_time, med_of_time

def name_remove_blank(name):
    n = name.split(" ")
    new_name = ""
    for nn in n:
        if nn != "":
            new_name += nn
            new_name += " "
    new_name = new_name[:-1]
    return new_name

def name_remove_non_alpha(name):
    for i in range(len(name)):
        if name[i].isalpha() == True:
            break
    return name[i:]

def filter_out(string):
    for i in range(len(string)):
        if string[i] == "<":
            break
    return string[:i]


"""
Sentiment analysis
"""
# https://github.com/shekhargulati/sentiment-analysis-python/blob/master/opinion-lexicon-English/positive-words.txt
def sentiment_words(file_name):
    # Using readlines()
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    
    sentiment_words = set()
    # Strips the newline character
    for line in Lines:
        if line[0] != ";":
            sentiment_words.add(line)
    return sentiment_words

"""
error and related functions
"""

def get_rank_of_res_true(res_true):
    dic_all = {}
    for key in res_true:
        act = deepcopy(res_true[key])
        act.sort(key=lambda act: act[-1])
        dic_all[key] = act[0][-3:]

    # return result, key is email, value is rank
    dic_res = {}
    rankings = sorted(dic_all.items(), key=lambda x: x[-1], reverse= True)
    for i, name in enumerate(rankings):
        dic_res[name[0]] = (i,name[1][:2]) # key two-path distance and minimum distance
    return dic_res

def ndcg_cal(rank_res_true, res_pred):
    keys = set(rank_res_true.keys()).intersection(res_pred.keys())
    if len(keys) != 0:
        tmp_res_pred = []
        for key in keys:
            tmp_res_pred.append([res_pred[key],key])
        sorted_res_pred = sorted(tmp_res_pred)
        new_res_pred = {}
        for i in range(len(sorted_res_pred)):
            new_res_pred[sorted_res_pred[i][1]] = i
        keys = list(keys)
        dcg, idcg = 0, 0
        for key in keys:
            rel_i = len(rank_res_true) - new_res_pred[key]
            pos_i = rank_res_true[key] + 1
            # rel_i = 1/(1+res_pred[key])
            # i = 1/(1+rank_res_true[key][0])
            if pos_i == 1:
                dcg += rel_i 
            else:
                dcg += (rel_i)/math.log(pos_i,2)
        for i in range(len(rank_res_true)):
            rel_i, pos_i = len(rank_res_true) - i, i + 1
            # rel_i, i = 1/(1+i), 1/(1+i)
            if pos_i == 1:
                idcg += rel_i 
            else:
                idcg += (rel_i)/math.log(pos_i,2)
        ndcg = dcg/idcg
        return ndcg
    else:
        return 0

def error_cal(res_pred, res_true, max_d = 10):
    dist = 0
    l = 0
    rank_res_true = get_rank_of_res_true(res_true)
    # rank_res_true[key] = (idx, [2-path num, dist])
    stat_pred = 0
    stat_true = [0, 0, 0, 0, 0] # 0-index not communicated before, 1-index communicated before, 2-index 2-path num, key not in res_pred, total key
    
    ndcgs = ndcg_cal(res_pred, res_true)

    for key in rank_res_true:
        l += 1
        if key in res_pred:
            dist += (res_pred[key] - rank_res_true[key][0]) ** 2
            stat_pred += 1
        else:
            dist += max_d ** 2
            if rank_res_true[key][1][1] == 1: # dist = 1, communicated before
                stat_true[1] += 1
            else: 
                stat_true[0] += 1
            if rank_res_true[key][1][0] != 0: # have 2-path 
                stat_true[2] += 1
            stat_true[3] += 1
    if l == 0:
        error = 0
    else:
        error = dist/l
    # deal with divide by zero
    if len(res_pred) == 0:
        a = 0
    else:
        a = stat_pred/len(res_pred)
    # if len(rank_res_true) == 0:
    #     b = np.array(stat_true)
    # else:
    #     b = np.array(stat_true)/len(rank_res_true)
    stat_true[4] = len(rank_res_true)
    b = np.array(stat_true)
    return error, a, b, ndcgs

def save_error(path, pre_s, err, err_by_date, error_by_date_sum, stat_pred, stat_true, ndcgs, ndcg_by_date):
    with open( path + pre_s + "test_error", "wb") as fp:   #Pickling
        pickle.dump(err, fp)
    with open( path + pre_s + "test_error_by_date", "wb") as fp:   #Pickling
        pickle.dump(err_by_date, fp)
    with open( path + pre_s + "test_error_by_date_sum", "wb") as fp:   #Pickling
        pickle.dump(error_by_date_sum, fp)
    with open( path + pre_s + "stat_true", "wb") as fp:   #Pickling
        pickle.dump(stat_true, fp)
    with open( path + pre_s + "stat_pred", "wb") as fp:   #Pickling
        pickle.dump(stat_pred, fp)
    with open( path + pre_s + "test_error_by_date_sum", "wb") as fp:   #Pickling
        pickle.dump(error_by_date_sum, fp)
    with open( path + pre_s + "ndcgs", "wb") as fp:   #Pickling
        pickle.dump(ndcgs, fp)
    with open( path + pre_s + "ndcg_by_date", "wb") as fp:   #Pickling
        pickle.dump(ndcg_by_date, fp)




"""
draw figure
"""
def draw_hist_seaborn(d, path, pre_s, idx_figure, s = ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'baseline','our']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    # 
    for i in range(len(d)):
        ax = sns.kdeplot(d[i],label= s[i],shade = True)
    ax.set_xlim(0,1)

    ax.set(title = 'Loss')
    plt.legend()
    filename = pre_s + "accuracy_all" + idx_figure + ".png"
    plt.savefig(path + "figure/enron/new/" + filename)
    plt.close()

def draw_curve_seaborn(d, path, pre_s, idx_figure, s = ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'baseline','our']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in range(len(d)):
        ax = plt.plot(d[i],label= s[i],alpha = 0.4)
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    plt.legend()
    filename = pre_s + "accuracy_all_curve" + idx_figure + ".png"
    plt.savefig(path + "figure/enron/new/" + filename)
    plt.close()


s = "UserDepartment UserJobTitle"
s = s.split()
for ss in s:
    print("# "+ss)
    print(ss + " = ")

for ss in s:
    print(ss)

print('return '+', '.join(s))