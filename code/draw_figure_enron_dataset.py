import pandas as pd
import json
from collections import defaultdict
import numpy as np
import networkx as nx
from grakel import kernels
from grakel.utils import graph_from_networkx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def get_name(email):
    try:
        all_list = email.split(", ")
        for i in range(len(all_list)):
            for j in range(len(all_list[i])):
                if all_list[i][j].isalpha():
                    break
            if j!=0:
                all_list[i] = all_list[i][j:]
        return all_list
    except:
        return []

def dictionary():
    return defaultdict(int)

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
def generate_json1(name = '2000and2001pattern.json'):
    csv_file = pd.read_csv(path + dataset[8])
    # 1999 start from 1138
    # 2000 start from 12282
    # 2001 start from 208383
    # 2002 start from 481415
    enron2000 = defaultdict(dictionary)
    email_set_2000 = set()
    enron2001 = defaultdict(dictionary)
    email_set_2001 = set()
    enron = {'2000': enron2000, '2001': enron2001}
    email_set = {'2000': email_set_2000, '2001': email_set_2001}
    for i in range(12282, len(csv_file)):
        sent_email = csv_file.iloc[i]['From']
        if 'enron' in sent_email:
            if '2000' in csv_file.iloc[i]['Date'] or '2001' in csv_file.iloc[i]['Date']:
                to_emails = get_name(csv_file.iloc[i]['To'])
                year = csv_file.iloc[i]['Date'][:4]
                for to_email in to_emails:
                    enron[year][sent_email][to_email] += 1
                    email_set[year].add(to_email)
                email_set[year].add(sent_email)
        if '2002' in csv_file.iloc[i]['Date']:
            break
    intersection_emails = email_set['2000'].intersection(email_set['2001'])
    print(len(intersection_emails))
    # for year in list(enron.keys()):
    #     for email in list(enron[year].keys()):
    #         if email not in intersection_emails:
    #             del(enron[year][email])
    #         for to_email in list(enron[year][email].keys()):
    #             if to_email not in intersection_emails:
    #                 del(enron[year][email][to_email])
    enron['200000'] = list(intersection_emails)
    json_pattern = json.dumps(enron)
    #name = '2000and2001pattern.json'
    f = open(path + name,"w")
    f.write(json_pattern)
    f.close()

def generate_json2(name = '400000prev_after.json'):
    csv_file = pd.read_csv(path + dataset[8])
    # 2000 start from 12282
    # 2001 start from 208383
    enron1 = defaultdict(dictionary)
    email_set1 = set()
    enron2 = defaultdict(dictionary)
    email_set2 = set()
    for i in range(300000, 400000):
        sent_email = get_name(csv_file.iloc[i]['From'])[0]
        if 'enron' in sent_email:   
            to_emails = get_name(csv_file.iloc[i]['To'])
            year = csv_file.iloc[i]['Date'][:4]
            for to_email in to_emails:
                enron1[sent_email][to_email] += 1
                email_set1.add(to_email)
            email_set1.add(sent_email)
    for i in range(400000, 500000):
        sent_email = get_name(csv_file.iloc[i]['From'])[0]
        if 'enron' in sent_email:   
            to_emails = get_name(csv_file.iloc[i]['To'])
            for to_email in to_emails:
                enron2[sent_email][to_email] += 1
                email_set2.add(to_email)
            email_set1.add(sent_email)
    intersection_emails = email_set1.intersection(email_set2)
    # print(len(intersection_emails))
    # for email in list(enron1.keys()):
    #     if email not in intersection_emails:
    #         del(enron1[email])
    #     for to_email in list(enron1[email].keys()):
    #         if to_email not in intersection_emails:
    #             del(enron1[email][to_email])
    # for email in list(enron2.keys()):
    #     if email == 'phillip.love@enron.com':
    #         print(1)
    #     if email not in intersection_emails:
    #         del(enron2[email])
    #     for to_email in list(enron2[email].keys()):
    #         if to_email not in intersection_emails:
    #             del(enron2[email][to_email])
    json_pattern = json.dumps([enron1,enron2, list(intersection_emails)])
    
    f = open(path + name,"w")
    f.write(json_pattern)
    f.close()

def dic2matrix(dic, keys):
    keys2idx = {}
    for idx,key in enumerate(keys):
        keys2idx[key] = idx
    matrix = np.zeros((len(keys), len(keys)))
    for from_email in dic.keys():
        if from_email in keys2idx:
            i = keys2idx[from_email]
            for to_email in dic[from_email].keys():
                if to_email in keys2idx:
                    j = keys2idx[to_email]
                    matrix[i][j] = dic[from_email][to_email]
    return matrix

def compute_laplacian(matrix):
    G = nx.from_numpy_array(matrix)
    a = nx.normalized_laplacian_spectrum(G)
    # print(a)
    # nx.draw(G)
    return a

def json2matrix(name = '400000prev_after.json', title = "400000.jpg", labels = ['Before Enron scandal happens', 'After Enron scandal happens']):
    f = open(path + name, "r")
    f = json.load(f)
    if type(f) is list:
        first = f[0]
        second = f[1]
        email_list = f[2]
    else:
        keys = list(f.keys())
        first = f[keys[0]]
        second = f[keys[1]]
        email_list = f[keys[2]]
    first_matrix = dic2matrix(first, email_list)
    second_matrix = dic2matrix(second, email_list)
    a1 = compute_laplacian(first_matrix)
    a2 = compute_laplacian(second_matrix)
    plt.plot(a1, label = labels[0])
    plt.plot(a2, label = labels[1])
    plt.legend()
    plt.xlabel('eigen index')
    plt.ylabel('eigenvalue') 
    plt.savefig(path + "figure/enron/final/" + title)

flag = 0
if flag == 0:
    # generate_json2()
    generate_json1(name = '2000and2001pattern.json')
else:
    json2matrix('2000and2001pattern.json',"two_year.jpg", ['2000', '2001'])