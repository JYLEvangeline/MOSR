from email.policy import default
from sys import set_asyncgen_hooks
from time import time
from typing import Dict
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from statsmodels.stats.weightstats import ztest as ztest
from dateutil import parser
# from torch import Set
import matplotlib.pyplot as plt
from statistics import mean
import json
import numpy as np
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import email
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from queue import Queue

def get_key(df):
    dic = {}
    for index, row in df.iterrows():
        dic[row['POST_ID']] = row['PROPERTIES']
    return dic

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
# dataset list
# 0 Bitcoin OTC trust weighted signed network
# 1 Social Network: Reddit Hyperlink Network
# 2 Social Network: Reddit Hyperlink Network
# 3 email-Eu-core temporal network
# 4 college message dataset
# 5 enron email dataset
from_set = defaultdict(list)
to_set = defaultdict(list)
ft_set = defaultdict(list)
# with open(path + dataset[3]) as f:
#     lines = f.readlines()
#     for line in lines:
#         string = line.replace("\n","").split(" ")
#         # making the smaller one before the larger one
#         if int(string[0]) > int(string[1]):
#             flag = -1
#             com_str = string[1]+'+'+string[0]
#         else:
#             flag = 1
#             com_str = string[0]+'+'+string[1]
#         ft_set[com_str].append([int(string[2]),flag])
#         from_set[string[0]].append([string[1],int(string[2])])
#         to_set[string[1]].append([string[0],int(string[2])])
# all_key = {}
# for key in ft_set.keys():
#     all_key[key] = [len(ft_set[key]),sum([t[1] for t in ft_set[key]])]
# print(1)
def draw_four_plots(data, title, suptitle):
    fig, axs = plt.subplots(2,2)
    fig.suptitle(suptitle)
    indices = [[0,0],[0,1],[1,0],[1,1]]
    for index, data_i, title_i in zip(indices,data,title):
        axs[index[0], index[1]].hist(data_i)
        axs[index[0], index[1]].set_title(title_i)     
        # axs[index[0], index[1]].set_ylim(0,50000) 
        # axs[index[0], index[1]].set_xlim(0,6) 
    plt.savefig(path + 'figure/'+suptitle+".png")

def draw_before_after_distribution(lines, time = 23708115):
    #before time point
    all_users = set()
    send_valid_users = set()
    send_user_before = []
    send_user_after = []

    to_valid_users = set()
    to_user_before = []
    to_user_after = []
    for line in lines:
        string = line.replace("\n","").split(" ")
        all_users.add(string[0])
        if int(string[2])< time-1:
            send_valid_users.add(string[0])
            send_user_before.append(int(string[2]))
            to_valid_users.add(string[0])
            to_user_before.append(int(string[2]))
    for line in lines:
        string = line.replace("\n","").split(" ")
        if int(string[2]) > time and string[0] in send_valid_users:
            send_user_after.append((int(string[2])-time))
        if int(string[2]) > time and string[1] in to_valid_users:
            to_user_after.append((int(string[2])-time))
            
    suptitle = 'Based on before time point'
    title = ['send_user_before','send_user_after','to_user_before','to_user_after']
    data = [send_user_before,send_user_after,to_user_before,to_user_after]
    draw_four_plots(data, title, suptitle)
    # after time point
    all_users = set()
    send_valid_users = set()
    send_user_before = []
    send_user_after = []

    to_valid_users = set()
    to_user_before = []
    to_user_after = []
    for line in lines:
        string = line.replace("\n","").split(" ")
        all_users.add(string[0])
        if int(string[2]) > time:
            send_valid_users.add(string[0])
            send_user_before.append((int(string[2])-time))
            to_valid_users.add(string[0])
            to_user_before.append((int(string[2])-time))
    for line in lines:
        string = line.replace("\n","").split(" ")
        if int(string[2]) < time+1 and string[0] in send_valid_users:
            send_user_after.append(int(string[2]))
        if int(string[2]) > time+1 and string[1] in to_valid_users:
            to_user_after.append(int(string[2]))
            
    suptitle = 'Based on after time point'
    title = ['send_user_before','send_user_after','to_user_before','to_user_after']
    data = [send_user_before,send_user_after,to_user_before,to_user_after]
    draw_four_plots(data, title, suptitle)

# generate a default form
def def_value():
    return [[],[]]

# get the difference for two lists in a list l  [[1,3,5],[2,8]] ->[2,2],[6],2,6
def get_diff(l):
    l0 = [l[0][i] - l[0][i-1] for i in range(1,len(l[0]))]
    l1 = [l[1][i] - l[1][i-1] for i in range(1,len(l[1]))]
    mean0 = mean(l0) if len(l0) != 0 else -1
    mean1 = mean(l1) if len(l1) != 0 else -1
    return l0,l1,mean0,mean1

def difference(lines,time = 23708115):
    from_users = defaultdict(def_value)
    to_users = defaultdict(def_value)
    print(time)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].append(cur_time)
            to_users[string[1]][0].append(cur_time)
        else:
            from_users[string[0]][1].append(cur_time)
            to_users[string[1]][1].append(cur_time)
    from_user_all = []
    to_user_all = []
    for key in from_users.keys():
        val = get_diff(from_users[key])
        # print(from_users[key])
        # print(val)
        # print("\\\\\\\\\\\\\\")
        try:
            from_user_all.append(ztest(val[0],val[1])[1])
        except:
            from_user_all.append(0)
    for key in to_users.keys():
        val = get_diff(to_users[key])
        try:
            to_user_all.append(ztest(val[0],val[1])[1])
        except:
            to_user_all.append(0)
    titles = ['from_distribution','to_distribution']
    fig, axs = plt.subplots(2)
    fig.suptitle('Z test for difference distribution')
    data = [from_user_all,to_user_all]
    for i in range(2):
        axs[i].hist(data[i],alpha=0.5,label = 'z test between before and after, time is '+str(time))
        axs[i].legend(loc='upper right')
        axs[i].set_title(titles[i])
    plt.savefig(path + 'figure/Z test for difference '+str(time)+'.png')
    plt.close()

# conduct the ztest
def difference_ztest(lines,time = 23708115):
    from_users = defaultdict(def_value)
    to_users = defaultdict(def_value)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].append(cur_time)
            to_users[string[1]][0].append(cur_time)
        else:
            from_users[string[0]][1].append(cur_time)
            to_users[string[1]][1].append(cur_time)
    from_user_all = []
    to_user_all = []
    for key in from_users.keys():
        val = get_diff(from_users[key])
        from_user_all[0].append(val[2])
        from_user_all[1].append(val[3])
    for key in to_users.keys():
        val = get_diff(to_users[key])
        to_user_all[0].append(val[2])
        to_user_all[1].append(val[3])
        
def between_distribution(lines, time = 23708115):
    def def_set():
        return [set(),set()]
    from_users = defaultdict(def_set)
    to_users = defaultdict(def_set)
    for line in lines:
        string = line.replace("\n","").split(" ")
        cur_time = int(string[2])
        if cur_time < time + 1:
            from_users[string[0]][0].add(string[1])
            to_users[string[1]][0].add(string[0])
        else:
            if string[1] not in from_users[string[0]][0]:
                from_users[string[0]][1].add(string[1])
            if string[0] not in to_users[string[1]][0]:
                to_users[string[1]][1].add(string[0])
    from_num, to_num = 0,0
    for key in from_users.keys():
        if len(from_users[key][1]) !=0:
            from_num +=1
    for key in to_users.keys():
        if len(to_users[key][1]) !=0:
            to_num +=1       
    return from_num, to_num
def between_distribution_change_with_time(lines, start_time = 0, final_time = 63708155, step = 500000):
    from_num_all, to_num_all = [],[]
    # final_time = 63708155
    # final_time = 1500000
    for i in range(start_time,final_time, 500000):
        from_num, to_num = between_distribution(lines,time = i)
        from_num_all.append(from_num)
        to_num_all.append(to_num)
    x = list(range(start_time,final_time, 500000))
    plt.plot(x,from_num_all,label = 'from')
    plt.plot(x,to_num_all,label = 'to')
    plt.xlabel('time')
    plt.ylabel('undiscovered relationships')
    plt.title('relationship growth')
    plt.legend()
    plt.savefig(path + 'figure/relationship growth.png')

def check_email_name(possible_emails,email_address):
    try:
        address = email_address.split(",")
    except:
        return
    # for pm in possible_emails:
    #     if pm not in address:
    for i in range(len(address)):
        try:
            address[i] = address[i].split("@")[1][:-2]
        except:
            address[i] = ''
    return address

def from_to_company(csv_file): # only for dataset 5
    possible_emails = ['hotmail','gmail','yahoo','enron']
    dic_from = defaultdict(int)
    dic_to = defaultdict(int)
    for i in range(len(csv_file)):
        from_email = check_email_name(possible_emails,csv_file.iloc[i]['From'])
        to_email = check_email_name(possible_emails,csv_file.iloc[i]['To'])
        if from_email:
            for email in from_email:
                dic_from[email] +=1
        if to_email:
            for email in to_email:
                dic_to[email] += 1
            
    json_from = json.dumps(dic_from)
    json_to = json.dumps(dic_to)

    f = open("from.json","w")
    f.write(json_from)
    f.close()
    f = open("to.json","w")
    f.write(json_to)
    f.close()

def get_name(email):
    try:
        all_list = email.split(", ")
        # for i in range(len(all_list)):
        #     all_list[i] = all_list[i][1:-1]
        return all_list
    except:
        return []
def get_enron_name(email_list):
    new_list = []
    for e in email_list:
        if 'enron' in e:
            new_list.append(e)
    return new_list
def from_to_name(csv_file): # only for dataset 5, 6
    possible_emails = ['hotmail','gmail','yahoo','enron']
    dic_from = defaultdict(int)
    dic_to = defaultdict(int)
    l = len(csv_file)
    l = 500
    for i in range(250000,300000):
        name_from = get_name(csv_file.iloc[i]['From'])
        name_to = get_name(csv_file.iloc[i]['To'])
        
        for name in name_from:
            dic_from[name] += 1
        for name in name_to:
            dic_to[name] += 1
        
        # except:
        #     print(1)
        #     name_from = get_name(csv_file.iloc[i]['From'])
        #     name_to = get_name(csv_file.iloc[i]['To'])
    
    json_from = json.dumps(dic_from)
    json_to = json.dumps(dic_to)

    f = open("from_name.json","w")
    f.write(json_from)
    f.close()
    f = open("to_name.json","w")
    f.write(json_to)
    f.close()
def emailformat():
    return defaultdict(list)
def get_stage(name):
    if name in dic_stage:
        return dic_stage[name]
    else:
        return 6
def intn1():
    return -1
def high_freqeuncy_name(csv_file):
    # all freqeuncy
    # from_name = [('kate.symes@enron.com', 5438), ('steven.kean@enron.com', 6759), ('tana.jones@enron.com', 8490), ('enron.announcements@enron.com', 8587), ('sara.shackleton@enron.com', 8777), ('chris.germany@enron.com', 8801), ('pete.davis@enron.com', 9149), ('jeff.dasovich@enron.com', 11411), ('vince.kaminski@enron.com', 14368), ('kay.mann@enron.com', 16735)]
    # to_name = [('paul.kaufman@enron.com', 8462), ('susan.mara@enron.com', 8860), ('pete.davis@enron.com', 9281), ('mark.taylor@enron.com', 9772), ('james.steffes@enron.com', 10176), ('sara.shackleton@enron.com', 11431), ('steven.kean@enron.com', 12641), ('tana.jones@enron.com', 12816), ('jeff.dasovich@enron.com', 14162), ('richard.shapiro@enron.com', 14665)]
    # name = set(['kate.symes@enron.com', 'steven.kean@enron.com', 'tana.jones@enron.com', 'sara.shackleton@enron.com', 'chris.germany@enron.com', 'pete.davis@enron.com', 'jeff.dasovich@enron.com', 'vince.kaminski@enron.com', 'kay.mann@enron.com'])
    
    # range from 250000-300000
    # from_name =  [('matthew.lenhart@enron.com', 712), ('tana.jones@enron.com', 792), ('sara.shackleton@enron.com', 794), ('steven.kean@enron.com', 1001), ('enron.announcements@enron.com', 1158), ('vince.kaminski@enron.com', 1750), ('jeff.dasovich@enron.com', 2018), ('kate.symes@enron.com', 2028), ('pete.davis@enron.com', 2458), ('kay.mann@enron.com', 2699)]
    # to_name = [('susan.mara@enron.com', 1550), ('harry.kingerski@enron.com', 1576), ('kate.symes@enron.com', 1595), ('sandra.mccubbin@enron.com', 1662), ('tana.jones@enron.com', 1762), ('paul.kaufman@enron.com', 1836), ('jeff.dasovich@enron.com', 1926), ('james.steffes@enron.com', 2117), ('richard.shapiro@enron.com', 2343), ('pete.davis@enron.com', 2459)]
    name = set(['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com'])
    dic_people = defaultdict(list)
    start_i = 0
    prev_day = parser.parse(csv_file.iloc[start_i]['Date']).day -1
    name_prev_day = defaultdict(intn1)
    name_rank = defaultdict(int)
    words = stopwords.words()
    for i in range(start_i,len(csv_file)):
        if i%10000 == 0:
            print(i)
        name_from = set(get_name(csv_file.iloc[i]['From']))
        name_to = set(get_name(csv_file.iloc[i]['To']))
        cur_day = parser.parse(csv_file.iloc[start_i]['Date']).day
        if len(name&name_from) > 0 or len(name&name_to) > 0:
            try:
                content_length, stop_length = remove_stop_words(csv_file.iloc[i]['Content'],words)
            except:
                content_length, stop_length = 1,0
        for nf in name&name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in name_to:
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                dic_people[nf].append((nt,stop_length,1,csv_file.iloc[i]['Date'],stop_length/content_length,nf_stage/nt_stage,name_rank[nf]))
                
        for nf in name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in name&name_to:
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                dic_people[nt].append((nf,stop_length,-1,csv_file.iloc[i]['Date'],stop_length/content_length,nt_stage/nf_stage,name_rank[nf]))
        
    json_people = json.dumps(dic_people)

    f = open(path + "chat.json","w")
    f.write(json_people)
    f.close()

    
def sort2(left,right):
    # keep left < right
    if left > right:
        return right, left
    return left, right

def online_two_path(graph,dic_two_path,new_edges):
    # update two_path
    for edges in new_edges:
        p1,p2 = edges.split(",")
        for p3 in graph[p1]: # p2-p1-p3
            if p3 not in graph[p2]:
                dic_two_path[p3][p2].add(p1)
                dic_two_path[p2][p3].add(p1)
        for p3 in graph[p2]: # p1-p2-p3
            if p3 not in graph[p1]:
                dic_two_path[p3][p1].add(p2)
                dic_two_path[p1][p3].add(p2)
        # graph[p1].add(p2)
        # graph[p2].add(p1)
    # update graph.  not update with two_path? count 1 day by 1 day? or use a ordered set?
    for edges in new_edges:
        p1,p2 = edges.split(",")
        graph[p1].add(p2)
        graph[p2].add(p1)
    return graph,dic_two_path

def dic_set():
    return defaultdict(set)

def shortest_path(graph, start, goal):
    explored = set()
     
    # Queue for traversing the
    # graph in the BFS
    queue = Queue()
    queue.put((start, [start]))
     
    # If the desired node is
    # reached
    if start == goal:
        return 0
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue.qsize()>0:
        (node, path) = queue.get()
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                queue.put((neighbour, path + [neighbour]))
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    return len(path) + 1
            explored.add(node)
    # Condition when the nodes
    # are not connected
    return -1


def high_freqeuncy_all(csv_file):
    # all freqeuncy
    # from_name = [('kate.symes@enron.com', 5438), ('steven.kean@enron.com', 6759), ('tana.jones@enron.com', 8490), ('enron.announcements@enron.com', 8587), ('sara.shackleton@enron.com', 8777), ('chris.germany@enron.com', 8801), ('pete.davis@enron.com', 9149), ('jeff.dasovich@enron.com', 11411), ('vince.kaminski@enron.com', 14368), ('kay.mann@enron.com', 16735)]
    # to_name = [('paul.kaufman@enron.com', 8462), ('susan.mara@enron.com', 8860), ('pete.davis@enron.com', 9281), ('mark.taylor@enron.com', 9772), ('james.steffes@enron.com', 10176), ('sara.shackleton@enron.com', 11431), ('steven.kean@enron.com', 12641), ('tana.jones@enron.com', 12816), ('jeff.dasovich@enron.com', 14162), ('richard.shapiro@enron.com', 14665)]
    # name = set(['kate.symes@enron.com', 'steven.kean@enron.com', 'tana.jones@enron.com', 'sara.shackleton@enron.com', 'chris.germany@enron.com', 'pete.davis@enron.com', 'jeff.dasovich@enron.com', 'vince.kaminski@enron.com', 'kay.mann@enron.com'])
    
    # range from 250000-300000
    # from_name =  [('matthew.lenhart@enron.com', 712), ('tana.jones@enron.com', 792), ('sara.shackleton@enron.com', 794), ('steven.kean@enron.com', 1001), ('enron.announcements@enron.com', 1158), ('vince.kaminski@enron.com', 1750), ('jeff.dasovich@enron.com', 2018), ('kate.symes@enron.com', 2028), ('pete.davis@enron.com', 2458), ('kay.mann@enron.com', 2699)]
    # to_name = [('susan.mara@enron.com', 1550), ('harry.kingerski@enron.com', 1576), ('kate.symes@enron.com', 1595), ('sandra.mccubbin@enron.com', 1662), ('tana.jones@enron.com', 1762), ('paul.kaufman@enron.com', 1836), ('jeff.dasovich@enron.com', 1926), ('james.steffes@enron.com', 2117), ('richard.shapiro@enron.com', 2343), ('pete.davis@enron.com', 2459)]
    name = set(['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com'])
    dic_people = defaultdict(list)
    start_i = 0
    prev_day = parser.parse(csv_file.iloc[start_i]['Date']).date()
    name_prev_day = defaultdict(lambda: datetime.min)
    name_rank = defaultdict(int)
    words = stopwords.words()
    exsiting_edges = set()
    edges_today = set()
    name_from_to = defaultdict(set)
    graph = defaultdict(set)
    dic_two_path = defaultdict(dic_set)
    distance = {}
    for i in range(start_i,len(csv_file)):
        print(i)
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
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                if key in distance:
                    dist = distance[key]
                else:
                    dist = shortest_path(graph, small,large)
                    distance[key] = dist
                dic_people[nf].append((nt,stop_length,1,csv_file.iloc[i]['Date'],stop_length/content_length,nf_stage/nt_stage,name_rank[nf],two_path_value, dist))
                
        for nf in name_from:
            if cur_day == name_prev_day[nf]:
                name_rank[nf] += 1
            else:
                name_rank[nf] = 1
            name_prev_day[nf] = cur_day
            for nt in enron_name_to:
                nt_stage = get_stage(nt)+1
                nf_stage = get_stage(nf)+1
                if nt in graph[nf]:
                    two_path_value = -1
                else:
                    two_path_value = len(dic_two_path[nf][nt])
                small, large = sort2(nf,nt)
                key = small + ',' + large
                if key in distance:
                    dist = distance[key]
                else:
                    dist = shortest_path(graph, small,large)
                    distance[key] = dist
                dic_people[nt].append((nf,stop_length,-1,csv_file.iloc[i]['Date'],stop_length/content_length,nt_stage/nf_stage,name_rank[nf],two_path_value, dist))
        for nf in name_from:
            for nt in name_to:
                name_from_to[nf].add(nt)
                if nf in name_from_to[nt]:
                    # smaller before
                    s = ','.join(sorted([nf,nt]))
                    edges_today.add(s)
        
    json_people = json.dumps(dic_people)

    f = open(path + "chat.json","w")
    f.write(json_people)
    f.close()

def get_pattern1(): # get pattern on percentage
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    l = []
    for key in dic.keys():
        count_hgy = 0
        count_enron = 0
        for s in dic[key]:
            for p in ['hotmail','gmail','yahoo']:
                if p in s[0]:
                    count_hgy += 1
            if 'enron' in s[0]:
                count_enron += 1
        length = len(dic[key])
        l.append(( count_enron/length, count_hgy/length, 1-count_enron/length- count_hgy/length))


def draw_seaborn(d, name):
    sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(path + "figure/enron/new/" + name+".png")

def draw_hist_seaborn(d, name, label = [0,0.5,1,2]):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in d.keys():
        ax = sns.kdeplot(d[i],label= 'ratio >= ' + str(i),shade = True)
    ax.set_xlim(0,1)
    # ax2 = sns.kdeplot(y,label='y',shade = True,palette="crest")
    # sns.kdeplot(data=d, x="total_bill", hue="size",shade = True, common_norm=False, palette="crest",alpha=.5, linewidth=0,)
    # # Compute the correlation matrix
    # corr = d.corr()

    # # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(11, 9))

    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,annot=True,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set(title = name)
    plt.legend()
    plt.savefig(path + "figure/enron/new/hist_" + name+".png")
    plt.close()
def remove_stop_words(text, words):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in words]
    return len(text_tokens),len(tokens_without_sw)

def get_pattern2(): # get pattern for each one
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    for key in dic.keys():
        l = [[],[],[],[]] # inside, outside, length, last reach time(by minute)    
        dic_tmp = {}
        for s in dic[key]:
            l[0].append(int('enron' in s[0]))
            l[1].append(int('enron' not in s[0]))
            l[2].append(s[1])
            time = parser.parse(s[3])
            if s[0] in dic_tmp:
                delta = time - dic_tmp[s[0]]
                delta = delta.days * 24 + delta.seconds/3600
            else:
                delta = 0
            dic_tmp[s[0]] = time
            l[3].append(delta)
        l = pd.DataFrame(data = l).T
        l.columns = ['inside','outside','length','time']
        name = key.split("@")[0]
        draw_seaborn(l, name)
        print(1)

def get_pattern3(): # get pattern for each one
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    #for key in dic.keys():
    for key in ['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com']:
        if key == 'kim.ward@enron.com':
            print(1)
        l = [[],[],[],[],[],[],[],[]] # inside, outside, length, length_ratio,last reach time(by minute), length of two path nums
        dic_tmp = {}
        for s in dic[key]:
            l[0].append(s[5]) # inside
            l[1].append(int('enron' not in s[0])) # outside
            l[2].append(s[1]) # length_remove_stop_words
            time = parser.parse(s[3])
            if s[0] in dic_tmp:
                delta = time - dic_tmp[s[0]]
                delta = delta.days * 24 + delta.seconds/3600
            else:
                delta = 0
            dic_tmp[s[0]] = time
            l[3].append(s[4]) # length_ratio
            l[4].append(delta) # reply time
            l[5].append(s[6]) # rank in a day
            l[6].append(s[-2]) # path num
            l[7].append(s[-1]) # shortest path

        l = pd.DataFrame(data = l).T
        l.columns = ['inside','outside','length','length_ratio','time','rank','pathnum','shortest path']
        name = key.split("@")[0]
        draw_seaborn(l, name)

def pattern4_bin_helper(stage, keys):
    res = []
    for i,key in enumerate(keys):
        if stage >= key:
            res.append(key)
    # if stage >=2:
    #     res.append(3)
    # if stage >=1:
    #     res.append(2)
    # if stage >=0.5:
    #     res.append(1)
    # res.append(0)
    return res
        

def get_pattern4(): # get pattern for stage vs length, stage vs time
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    for key in ['kenneth.lay@enron.com','mike.grigsby@enron.com','kam.keiser@enron.com','marie.heard@enron.com','stephanie.panus@enron.com','richard.shapiro@enron.com','john.arnold@enron.com','kim.ward@enron.com']:
        dic_tmp = {}
        histgram_length = {0:[],1:[],2:[],3:[]} # 0,0.5,1,2
        histgram_time =  {0:[],1:[],2:[],3:[]}
        for s in dic[key]:
            time = parser.parse(s[3])
            if s[2] == 1:
                res_stage = pattern4_bin_helper(1/s[5])
                if s[0] in dic_tmp:
                    delta = time - dic_tmp[s[0]]
                    delta = delta.days * 24 + delta.seconds/3600
                else:
                    delta = 0
                for r in res_stage:
                    histgram_length[r].append(s[4])
                    histgram_time[r].append(delta)  
            else:
                dic_tmp[s[0]] = time
        draw_hist_seaborn(histgram_length, 'length_'+key)  
        draw_hist_seaborn(histgram_time, 'time_'+key)   


def get_pattern5(): # get pattern for stage vs length, stage vs time all together
    f = open(path + "chat.json",'r')
    dic = json.load(f)
    keys = [0, 0.5, 1, 2, 5]
    histgram_length, histgram_time = {},{}
    for key in keys:
        histgram_length[key] = [] # 0,0.5,1,2
        histgram_time[key] = []
    for key in dic.keys():
        dic_tmp = {}  
        for i,s in enumerate(dic[key]):
            time = parser.parse(s[3])
            if s[2] == 1:
                res_stage = pattern4_bin_helper(1/s[5], keys)
                if s[0] in dic_tmp:
                    delta = time - dic_tmp[s[0]]
                    delta = delta.days * 24 + delta.seconds/3600
                else:
                    delta = 0
                for r in res_stage:
                    histgram_length[r].append(s[4])
                    if delta != 0:
                        histgram_time[r].append(delta)  
            else:
                dic_tmp[s[0]] = time
    draw_hist_seaborn(histgram_length, 'length_all', keys)  
    draw_hist_seaborn(histgram_time, 'time_all', keys)   

# time_distribution = []
# data[3] 23708115
# data[4] mid 1085119730 min 1082040961 max 1098777142
# with open(path + dataset[4]) as f:
#     lines = f.readlines()
# my_dict = { 'name' : ["a", "b", "c", "d", "e","f", "g"],\
#                    'age' : [20,27, 35, 55, 18, 21, 35],\
#                    'designation': ["VP", "CEO", "CFO", "VP", "VP", "CEO", "MD"]}
# df = pd.DataFrame(my_dict)
# df.to_csv('csv_example.csv',index=False)
# df_csv = pd.read_csv('csv_example.csv')
# print(1)
# df_csv.sort_values(["age"],axis=0, ascending=True,inplace=True,na_position='first')
# df_csv.to_csv('test.csv',index=False)
# new_csv = pd.read_csv('test.csv')
# print(1)


# # start dataset 6
# csv_file = pd.read_csv(path + dataset[6])
# get_pattern2()


# start dataset 7
# csv_file = pd.read_csv(path + dataset[8])
# dic = {}
# l = set()
# keys = email.message_from_string(csv_file.iloc[0]['message']).keys()
# dic = defaultdict(list) 
# for message in csv_file['message']:
#     e = email.message_from_string(message)
#     for key in keys:
#         dic[key].append(e[key])
#     m = message.split("\n\n")
#     dic['Content'].append(''.join(m[1:]))
    
# dic['file'] = list(csv_file['file'])
# df = pd.DataFrame(dic)
# df['Date'] = pd.to_datetime(df['Date'])
# df.sort_values(["Date"],axis=0, ascending=True,inplace=True,na_position='first')
# df.to_csv(path+'sorted_emails.csv',index=False)


# start dataset 8

csv_file = pd.read_csv(path + dataset[8])
org = pd.read_csv(path + 'organazition2.csv')
dic_stage = {}
for i in range(len(org)):
    dic_stage[org.iloc[i]['Email']] = org.iloc[i]['Stage'] 

# resort by date
# csv_file['Date'] = pd.to_datetime(csv_file['Date'])
# csv_file.sort_values(["Date"],axis=0, ascending=True,inplace=True,na_position='first')
# csv_file.to_csv(path+'sorted_emails1.csv',index=False)


high_freqeuncy_all(csv_file)
get_pattern3()
# get_pattern3()
# title = pd.read_excel(path+'Enron_Employee_Status.xls',index_col=None)
# title_iloc = {}
# import pickle
# a = open(path+"data.pkl",'rb')
# dic = pickle.load(a)
# email = []
# for i in range(len(title)):
#     s = list(dic[title.iloc[i]['Name'].lower()])[0]
#     try:
#         s = s.split()[0]
#     except:
#         s = ""
#     email.append(s)
    
# print(1)
# # waiting = ['timothyheizenrader', 'cristopher foster', 'christopher gaskill', 'bill rapp ', 'vince j kaminski', 'john lloldra', 'shelley corman ', 'andrea ring', 'berney aucoin', 'jeffery skilling', 'bradley mckay', ' judy townsend', 'keith holst ', 'kimberly watson ', 'geoffery storey', 'peter keavey', 'carol clair', 'susan pereira', 'thomas martin', 'mark haedicke', 'albert meyers  ', 'jeffrey a shankman', 'eric lynder', 'andrew lewis', 'craig dean', 'harpreet arora', 'jeffery richter', 'monika causholli ', 'paul lucci ', 'susan bailey ', 'john zufferli ', 'david delainey', 'daron giron', 'drew fossum ', 'timothy belden', 'daren farmer', 'richard sanders', 'chris germany ', 'stacey white   ', 'laura luce ', 'douglas gilberth-smith', 'stacy dickson', 'michael grigsby', 'jane tholt', 'paul thomas ', 'patrice mims', 'christopher calger', 'cooper richey ', 'stanley horton ', 'thomos alonso', 'holden salisbury ', 'micheal swerzzbin', 'steven j kean', 'charles weldon', 'hunter shively', 'lawrence may', 'kim ward ', 'kevin presto', 'jay reitmeyer ', 'lindy donoho ', 'cara semperger  ', 'danny mccarty ', 'phillip m love', 'joe parks  ', 'michael curry', 'tori kuykendall ', 'randall gay', 'james schweiger', 'steve wang', 'james steffes', 'errol mclaughlin ', 'john lavorato', 'dan hyvl', 'michael maggi', 'john forney', 'jacob thomas', 'fletcher sturm', 'jason wolfe ', 'philip allen', 'richard ring', 'matthew motley', 'mark davis', 'john giffith', 'sandra brawner', 'frank vickers', 'jim schwieger ']


# # title_email = defaultdict(set)
# # csv_file = pd.read_csv(path + dataset[8])
# # dic = defaultdict(set)
# # for i in range(len(csv_file)):
# #     if i %5000 == 0:
# #         print(i)
# #     try:
# #         a  = csv_file.iloc[i]['X-From']
# #         for name in name_set:
# #             if name in csv_file.iloc[i]['X-From'].lower():
# #                 title_email[csv_file.iloc[i]['X-From'].lower()].add(csv_file.iloc[i]['From']+' '+csv_file.iloc[i]['X-From'].lower())
# #     except:
# #         a = 2
# # waiting = []
# # for name in name_set:
# #     if name not in title_email.keys():
# #         waiting.append(name)
# # wrong_value = []
# # for key,value in title_email.items():
# #     if len(value) > 1:
# #         wrong_value.append(key)
# # waiting = ['timothyheizenrader', 'cristopher foster', 'christopher gaskill', 'bill rapp ', 'vince kaminski', 'john lloldra', 'shelley corman ', 'andrea ring   ', 'berney aucoin', 'jeffery skilling', 'bradley mckay', ' judy townsend', 'keith holst ', 'kimberly watson ', 'geoffery storey', 'peter keavey', 'carol clair', 'susan pereira', 'thomas martin', 'mark haedicke', 'albert meyers  ', 'jeffrey shankman', 'eric lynder', 'andrew lewis', 'craig dean', 'harpreet arora', 'jeffery richter', 'monika causholli ', 'paul lucci ', 'susan bailey ', 'john zufferli ', 'david delainey', 'daron giron', 'drew fossum ', 'timothy belden', 'daren farmer', 'richard b sanders', 'chris germany ', 'stacey white   ', 'laura luce ', 'douglas gilberth-smith', 'stacy dickson', 'michael grigsby', 'jane tholt', 'paul thomas ', 'patrice mims', 'christopher calger', 'cooper richey ', 'stanley horton ', 'thomos alonso', 'holden salisbury ', 'micheal swerzzbin', 'steven kean', 'charles weldon', 'hunter shively', 'lawrence may', 'kim ward ', 'kevin presto', 'jay reitmeyer ', 'lindy donoho ', 'cara semperger  ', 'danny mccarty ', 'phillip love', 'joe parks  ', 'michael curry', 'tori kuykendall ', 'randall gay', 'james schweiger', 'steve wang', 'james steffes', 'errol mclaughlin ', 'john lavorato', 'dan hyvl', 'michael maggi', 'john forney', 'jacob thomas', 'fletcher sturm', 'jason wolfe ', 'philip allen', 'richard ring', 'matthew motley', 'mark davis', 'john giffith', 'sandra brawner', 'frank vickers', 'jim schwieger ']
# # waiting2 = ['timothyheizenrader', 'foster', 'gaskill', 'rapp', 'kaminski', 'lloldra', 'corman', 'ring', 'aucoin', 'skilling', 'mckay', 'townsend', 'holst', 'watson', 'storey', 'keavey', 'clair', 'pereira', 'martin', 'haedicke', 'meyers', 'shankman', 'lynder', 'lewis', 'dean', 'arora', 'richter', 'causholli', 'lucci', 'bailey', 'zufferli', 'delainey', 'giron', 'fossum', 'belden', 'farmer', 'sanders', 'germany', 'white', 'luce', 'gilberth-smith', 'dickson', 'grigsby', 'tholt', 'thomas', 'mims', 'calger', 'richey', 'horton', 'alonso', 'salisbury', 'swerzzbin', 'kean', 'weldon', 'shively', 'may', 'ward', 'presto', 'reitmeyer', 'donoho', 'semperger', 'mccarty', 'love', 'parks', 'curry', 'kuykendall', 'gay', 'schweiger', 'wang', 'steffes', 'mclaughlin', 'lavorato', 'hyvl', 'maggi', 'forney', 'thomas', 'sturm', 'wolfe', 'allen', 'ring', 'motley', 'davis', 'giffith', 'brawner', 'vickers', 'schwieger']
# # print(title_email)
# # import pickle
# # a_file = open(path + "data.pkl", "rb")
# # a = pickle.load(a_file)
# # a_file.close()

# # waiting_origin = ['skilling', 'ward', 'storey', 'arora', 'foster', 'causholli', 'schwieger', 'luce', 'giffith', 'grigsby', 'gilberth-smith', 'lynder', 'ring', 'germany', 'richter', 'wang', 'salisbury', 'bailey', 'mclaughlin', 'lucci', 'gaskill', 'mccarty', 'donoho', 'giron', 'aucoin', 'mckay', 'schweiger', 'thomas', 'alonso', 'holst', 'motley', 'belden', 'thomas', 'reitmeyer', 'fossum', 'richey', 'white', 'may', 'kuykendall', 'dean', 'meyers', 'zufferli', 'allen', 'townsend', 'swerzzbin', 'ring', 'parks', 'semperger', 'horton', 'watson', 'timothyheizenrader', 'rapp', 'maggi', 'wolfe', 'curry', 'corman', 'lloldra', 'weldon']
# waiting = ['charles weldon', 'thomos alonso', 'albert meyers', 'micheal swerzzbin', 'matthew motley', 'timothy belden', 'berney aucoin', 'james schweiger', 'bill rapp ', 'jason wolfe', 'michael maggi', 'steve wang', 'john giffith', 'harpreet arora', 'daron giron', 'eric lynder', 'joe parks', 'michael curry', 'timothy heizenrader', 'craig dean', 'bradley mckay', 'jeffery richter', 'keith holst', 'michael grigsby', 'richard ring', 'douglas gilberth-smith', 'philip allen', 'jacob thomas', 'lawrence may', ' judy townsend', 'jeffery skilling', 'john lloldra', 'christopher gaskill', 'geoffery storey', 'cristopher foster']
# a = [aa.split()[-1] for aa in waiting]
# dic = {}
# for aa,ww in zip(a,waiting):
#     dic[aa] = ww
# name_set = set(a)
# title_email = defaultdict(set)
# csv_file = pd.read_csv(path + dataset[8])
# n = ['lloldra']
# p = ['john']
# for i in range(len(csv_file)):
#     if i %5000 == 0:
#         print(i)
#     # try:
#     #     a  = csv_file.iloc[i]['X-From'].lower()
#     #     aa  = a.split()
#     #     name = aa[-1]
#     #     if 'thomas' in csv_file.iloc[i]['From']:
#     #         print(1)
#     # except:
#     #     a = 2
#     for nn,pp in zip(n,p):
#         try:
#             if nn in csv_file.iloc[i]['From']:
#                 j = csv_file.iloc[i]['From'].find(nn)
#                 s = csv_file.iloc[i]['From'][j-20:j+20]
#                 if 'enron' in s:
#                     print(nn,pp, s)
#             if nn in csv_file.iloc[i]['To']:
#                 j = csv_file.iloc[i]['To'].find(nn)
#                 s = csv_file.iloc[i]['To'][j-20:j+20]
#                 if 'enron' in s:
#                     print(nn,pp, s)
#         except:
#             a = 3
# waiting = []
# for name in name_set:
#     if name not in title_email.keys():
#         waiting.append(name)
# wrong_value = []
# for key,value in title_email.items():
#     if len(value) > 1:
#         wrong_value.append(key)
# print(1)

# csv_file.sort_values(["Date"],axis=0, ascending=True,inplace=True,na_position='first')
# csv_file.to_csv('sorted_enron.csv',index=False)
# q = pd.read_csv('sorted_enron.csv')
# print(q)
# draw_before_after_distribution(lines, time = 1085119730)

# between_distribution_change_with_time(lines)
# cur_dataset = dataset[1]
# dic_idx2name = {}
# with open(path + cur_dataset[:-3]+'txt') as f:
#     lines = f.readlines()
#     for line in lines:
#         string = line.split(". ")
#         dic_idx2name[int(string[0])-1] = string[1].replace("\n","")


# df1 = pd.read_csv(path + cur_dataset,sep = '\t')
# source = df1['SOURCE_SUBREDDIT']
# target = df1['TARGET_SUBREDDIT']
# c = Counter(target)
# keys_target = set()
# for key in c:
#     if c[key] > 9:
#         keys_target.add(key)
#         keys_target = set()
# c = Counter(source)
# source_target = set()
# for key in c:
#     if c[key] > 9:
#         source_target.add(key)
# source_dic = defaultdict(set)
# max_len = 0
# max_num = None
# # for i in range(len(df1)):
# for i in range(500):
#     source_dic[df1.iloc[i]['SOURCE_SUBREDDIT']].add(df1.iloc[i]['PROPERTIES'])
#     if len(source_dic[df1.iloc[i]['SOURCE_SUBREDDIT']]) > max_len:
#         max_len = len(source_dic[df1.iloc[i]['SOURCE_SUBREDDIT']])
#         max_num = df1.iloc[i]
# print(1)
# idx = []
# string  = 'hailcorporate'
# for i in range(500):
#     if df1.iloc[i]['SOURCE_SUBREDDIT'] == string:
#         idx.append(i)
# print(1)
# dic2 = get_key(df2)
# print(dic1==dic2)
# focus on df1
# dic = defaultdict(int)
# for i in range(len(df1)):
#     pro = df1.iloc[i]['PROPERTIES'].split(",")
#     for j,t in enumerate(pro):
#         if float(t)!= 0:
#             dic[dic_idx2name[j]] += 1
# import csv
# w = csv.writer(open(path+"output.csv", "w"))
# # 18,19,20 pos/neg/compound
# # loop over dictionary keys and values
# for key, val in dic.items():
#     # write every key and value to file
#     w.writerow([key, val])