# break users into stationary vs non-stationary groups. For every time step, how many people is larger than 2. 
# 2001.10.1

# this is the final version!!!!!!
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
from train_test_differently_MSOR import Model
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

def build_test_set(start_date, delta, dic, csv_file, dic_stage):
    end_date = start_date + delta
    if start_date not in dic:
        iloc = get_date_iloc(csv_file, start_date)
        dic[start_date] = iloc
    start_iloc = dic[start_date]
    if end_date not in dic:
        iloc = get_date_iloc(csv_file, end_date, start_i = start_iloc)
        dic[end_date] = iloc
    end_iloc = dic[end_date]
    # start_iloc, end_iloc = 0, 3
    json_people = high_frequency_iloc(csv_file, start_iloc, end_iloc, dic_stage)
    return json_people, dic
     

def high_frequency_iloc(csv_file, start_iloc, end_iloc, dic_stage):
    dic_people = defaultdict(list)
    prev_day = parser.parse(csv_file.iloc[start_iloc]['Date']).date()
    name_prev_day = defaultdict(lambda: datetime.min)
    name_rank = defaultdict(int)
    words = stopwords.words()
    exsiting_edges = set()
    edges_today = set()
    name_from_to = defaultdict(set)
    graph = defaultdict(set)
    dic_two_path = defaultdict(dic_set)
    distance = {}
    for i in range(start_iloc,end_iloc + 1):
    # for i in range(start_i,end_i):
        if i% 10000 == 0:
            print(1)
        # deal with X-cc
        try:
            math.isnan(csv_file.iloc[i]['X-cc']) == False
            cc_receiver = []
        except:
            # print(csv_file.iloc[i]['X-cc'])
            a = re.findall(r"CN=(.*?)>", csv_file.iloc[i]['X-cc'])
            cc_receiver = [aa.split("CN=")[-1].lower()+"@enron.com" for aa in a]
            if a == []:
                a = re.findall(r"<(.*?)>", csv_file.iloc[i]['X-cc'])
                cc_receiver = [aa.lower() for aa in a]
                if a == []:
                    # print(csv_file.iloc[i]['X-cc'])
                    a = re.findall(r"'(.*?)'", csv_file.iloc[i]['X-cc'])
                    cc_receiver = [aa.lower() for aa in a]
                    if a == []:
                        a = csv_file.iloc[i]['X-cc'].split(",")
                        cc_receiver = [aa.lower() for aa in a]
                        if a == []:
                            print(1)
                else:
                    if "list not shown" in a[0]:
                        cc_receiver = []
            
        if i%10000 == 0:
            print(i)
        name_from = set(get_name(csv_file.iloc[i]['From']))
        enron_name_from = get_enron_name(name_from)
        name_to = set(get_name(csv_file.iloc[i]['To']))
        enron_name_to = get_enron_name(name_to)
        cur_day = parser.parse(csv_file.iloc[i]['Date']).date()
        if cur_day != prev_day:
            new_edges = edges_today - exsiting_edges
            graph, dic_two_path = online_two_path(graph,dic_two_path,new_edges)
            exsiting_edges.update(new_edges)
            edges_today = set()
            prev_day = cur_day
            distance = {}
        if len(enron_name_from) > 0 or len(enron_name_to) > 0:
            try:
                content_length, stop_length = remove_stop_words(csv_file.iloc[i]['Content'],words)
            except:
                content_length, stop_length = 1,0
        for nf in enron_name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in name_to:
                nt_stage = get_stage(nt, dic_stage)+1
                nf_stage = get_stage(nf, dic_stage)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                # if key in distance: # jiayi distance is wrong needs to be updated?
                #     dist = distance[key]
                # else:
                #     dist = shortest_path(graph, small,large)
                #     distance[key] = dist
                dist = 0
                # if nt in graph[nf] or nf in graph[nt]:
                #     dist = 1
                # else:
                #     dist = -1
                dic_people[nf].append((nt,stop_length,1,csv_file.iloc[i]['Date'],stop_length/content_length,nf_stage/nt_stage,name_rank[nf],two_path_value, dist, cc_receiver))
                
        for nf in name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in enron_name_to:
                nt_stage = get_stage(nt, dic_stage)+1
                nf_stage = get_stage(nf, dic_stage)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                # if key in distance:
                #     dist = distance[key]
                # else:
                #     dist = shortest_path(graph, small,large)
                #     distance[key] = dist
                if nt in graph[nf] or nf in graph[nt]:
                    dist = 1
                else:
                    dist = -1
                dic_people[nt].append((nf,stop_length,-1,csv_file.iloc[i]['Date'],stop_length/content_length,nt_stage/nf_stage,name_rank[nf],two_path_value, dist, cc_receiver))
        for nf in name_from:
            for nt in name_to:
                a,b = sorted([nf,nt])
                if a in name_from_to[b]:
                    s = ','.join(sorted([nf,nt]))
                    edges_today.add(s)
                else:
                    name_from_to[b].add(a)
                    # smaller before
    json_people = json.dumps(dic_people)
    return json_people

def start_date_to_str(start_date):
    date_in_list = map(str, [start_date.year, start_date.month, start_date.day])
    return '.'.join(date_in_list)

def construct_matrix_dic(dic1, dic2, keys):    
    diff_dic = {}
    i = 0
    for key in keys:
        i +=1 
        if i%10000==0:
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
        elif val > seperate_val:
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
    json_non_stationary = json.dumps(dic_non_stationary)
    json_stationary = json.dumps(dic_stationary)
    return json_non_stationary, json_stationary

def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("--max_d", "-md", type=int, default = 10, help="Int parameter")
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0)
    parser.add_argument("-part_or_all", "-poa", type = str, default= 'all')
    # parser.add_argument("-train_year", "-trainy", type = int, default = 2000)
    # parser.add_argument("-test_year","-testy",type = int, default = 2000)
    parser.add_argument("-start_date", "-sd", type = str, default = "2001.10.1")
    parser.add_argument("-train_length","-trainl", type = int, default = 120) # days
    parser.add_argument("-seperate_val","-sv", type = int, default = 1) # days
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
    dic = {}
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
        file_path = path + 'break_into_consistent/MSOR3/' + start_date_to_str(start_date) + "delta" + str(args.train_length) + ".json"
        if os.path.exists(file_path) == False:
            # create json file for current time period
            json_people, dic = build_test_set(start_date, delta, dic, csv_file, dic_stage)
            # f = open(file_path,"w")
            # f.write(json_people)
            # f.close()
            open(file_path, "w").write(json_people)

        json_people = json.load(open(file_path,"r"))
        all_json.append(json_people)
        if i == 1:
            start_date += relativedelta(months=+3)
    # json file at the last time period in train
    json_in_train, json_name_in_train = all_json[0], json_names[0]
    keys_in_train = set(json_in_train.keys())
    # json files in test
    all_json, json_names = all_json[1:], json_names[1:]
    
    stationary_or_not_stats = []
    print(json_names)
    
    results = defaultdict(list)
    for json_in_test, json_name_in_test in zip(all_json, json_names):
        file_path = path + 'break_into_consistent/MSOR3/' + json_name_in_train + "vs" + json_name_in_test + str(args.train_length) + ".json"
        if os.path.exists(file_path) == False:
            keys_in_test = set(json_in_test.keys())
            keys = keys_in_train.intersection(keys_in_test)
            keys = list(keys)
            diff_dic = construct_matrix_dic(json_in_train, json_in_test, keys)
            json_diff_dic = json.dumps(diff_dic)
            open(file_path, "w").write(json_diff_dic)
        json_diff_dic = json.load(open(file_path,"r"))
        stationary_or_not_stats+= list(json_diff_dic.values())
        # calculate results
        # read MSOR model
        model = pickle.load(open("../data/model/differently/our/distance2_all_num_version0_max_d10_two_path_num_learning_rate0.991999.1.11004.model","rb"))
        res_without_change = model.test_without_change(json_in_test, start_date-relativedelta(months=20), start_date) # start_date-relativedelta(months=20), start_date can be changed to any wider range
        res_with_change = model.test(json_in_test, start_date-relativedelta(months=20), start_date) # start_date-relativedelta(months=20), start_date can be changed to any wider range
        results[json_name_in_test + '_without_change'] = res_without_change
        results[json_name_in_test + '_with_change'] = res_with_change
    print(stationary_or_not_stats)
    print(json_names)
    file_path = path + 'break_into_consistent/MSOR_results3' + "seperate_val" + str(args.seperate_val)
    with open( file_path, "wb") as fp:   #Pickling
        pickle.dump(results, fp)
    
    
        
    

if __name__ == "__main__":
    main()