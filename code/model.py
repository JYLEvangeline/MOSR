from collections import defaultdict
from datetime import datetime, timedelta
from dateutil import parser

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

def init_weights():
    return [1,1]

class Model:
    def __init__(self, weights, alphas, time_window_closeness, time_window_timeliness, owa_parameter, distance_measure="two_path_num", k = 3):
        """initilization

        Args:
            weights (_type_): _description_
            alphas (_type_): _description_
            time_window_closeness (_type_): _description_
            distance_measure (_type_): _description_
        """
        self.weights = weights
        self.alphas = alphas
        # time window for closeness
        self.time_window_closeness = time_window_closeness
        # time window for timeliness
        self.time_window_timeliness = time_window_timeliness
        # the parameter for OWA
        self.owa_parameter = owa_parameter
        # distance measure methods
        self.distance_measure = distance_measure
        self.k = k
        # initilize OWA weights
        self.Q_function()
        # initilize our model weights 
        self.user_weights = {}
        # initilize exisiting functions
        self.learning_functions = [self.OWA1, self.OWA2, self.timeline]
        self.learning_rate = 0.99

    # def insider(self, s, dic_followup):
    #     time = parser.parse(s[3])
    #     if s[0] in dic_followup:
    #         delta = time - dic_followup[s[0]]
    #         delta = (delta.days * 24 + delta.seconds/3600)/24
    #     else:
    #         delta = 0
    #     dic_followup[s[0]] = time

    # def ousider(self, chat_history,i,time_window):
    #     time = parser.parse(chat_history[i][3])
    #     self.frequency(chat_history[:i], time-time_window)

    def distance(self,chat_history_i):
        if self.distance_measure == "two_path_num":
            return math.exp(chat_history_i[7])
        elif self.distance_measure == "shortest_path":
            return math.exp(chat_history_i[8])
        else:
            return 1

    def frequency(self, chat_history, time_stamp):
        """get the frequency 

        Args:
            chat_history (list): the partial chat history between e_i and e_j
            time_stamp (datetime): t_i - w.
        Returns:
            the to/from frequency
        """
        # chat_history[i][2] indicate the direction. -1 is 'from' (e_j->e_i). 1 is 'to' (e_i->e_j).
        f = {-1:0,1:0}
        for i in range(len(chat_history)-1,-1,-1):
            cur_time = parser.parse(chat_history[i][3])
            if cur_time >= time_stamp:
                f[chat_history[i][2]] += 1
            else:
                break
        f = list(f.values())
        gamma_frequency = math.exp(sum([f_i*weight_i for f_i, weight_i in zip(f, self.weights)]))
        return gamma_frequency
    
    def closeness(self, chat_history, flag):
        """get the closeness

        Args:
            chat_history (list): the chat history between e_i and e_j
            i: the index of current chat history
            
            flag (int): indicates insider closeness or outsider closeness
            weights (list): w_1 and w_2. weights for to-frequency and from-frequency. 
        """
        cur_time = parser.parse(chat_history[-1][3])
        # time_window (datetime): w. default 1 day. 
        frequency = self.frequency(chat_history[:-1],cur_time - self.time_window_closeness)
        # insider
        if flag == True:
            stage_ratio = math.exp((0.5/chat_history[-1][5]))
            return frequency * stage_ratio
        else:
            distance = self.distance(chat_history[-1])
            return frequency * distance
    
    def time(self, content, flag):
        if flag == 1:
            dic = self.follow_up_time
            time_window_timeliness = self.time_window_timeliness
        else:
            dic = self.reply_time
            time_window_timeliness = 0
        time = parser.parse(content[3])
        if content[0] in dic:
            delta = time - dic[content[0]]
            delta = (delta.days * 24 + delta.seconds/3600)/24
        else:
            delta = 0
        delta = delta if delta > time_window_timeliness else 0
        return math.exp(delta)

    def timeliness(self,content):
        follow_up_time = self.time(content,1)
        reply_time = self.time(content,-1)
        xi_timeliness = self.alphas[0] * follow_up_time + self.alphas[1] * reply_time
        if follow_up_time != 0:
            self.time(content,1)
        if reply_time != 0:
            self.time(content,-1)
        return xi_timeliness


    def exp(self,p,alpha):
        return math.pow(p,alpha)
    
    def pow(self,p,alpha):
        return math.pow(alpha,p)

    def constant(self,p,alpha):
        return p

    def Q_function(self, type = "exp"):
        if type == "exp": # p ^ \alpha
            func = self.exp
        elif type == 'pow': # \alpha ^ p
            func = self.pow
        else:
            func = self.constant
        self.Q_funcs = []
        for i in range(self.k):
            self.Q_funcs.append(func(i+1, self.owa_parameter)-func(i, self.owa_parameter))

    def OWA1(self, one_day_activity):
        scores = []
        for act in one_day_activity:
            sorted_act = sorted(act[:self.k])
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        return rankings
    
    def OWA2(self, one_day_activity):
        scores = []
        for act in one_day_activity:
            sorted_act = sorted(act[:self.k])
            scores.append(sum([Q * a for Q, a in zip(self.Q_funcs, sorted_act)]))
        rankings = np.argsort(np.array(scores))
        return rankings[::-1]

    def timeline(self,one_day_activity):
        rankings = [[], []]
        follow_up_time, reply_time = [],[]
        for i, act in enumerate(one_day_activity):
            follow_up_time.append((act[3], i))
            reply_time.append((act[4], i))
        follow_up_time.sort(reverse = True)
        reply_time.sort(reverse = True)
        rankings[0] = [t[1] for t in follow_up_time]
        rankings[1] = [t[1] for t in reply_time]
        return np.array(rankings)[1]
    
    def updates(self,errors):
        # delta_weights = np.zeros(len(self.learning_functions))
        # if 0 not in errors:
        #     for i, error in enumerate(errors):
        #         delta_weights[i] = 1/error
        # else:
        #     for i, error in enumerate(errors):
        #         if error == 0:
        #             delta_weights[i] = 1
        # sum_delta_weight = sum(delta_weights)
        # for i, delta_weight in enumerate(delta_weights):
        #     delta_weights[i] = delta_weight/sum_delta_weight
        # return delta_weights
        delta_weights = np.zeros(len(self.learning_functions))
        index = errors.index(min(errors))
        delta_weights[index] = 1
        return delta_weights

    def our_model(self,one_day_activity, cur_user):
        rankings = []
        errors = []
        for func in self.learning_functions:
            func_res = func(one_day_activity)
            func_err = self.error(func_res,range(len(func_res)))
            rankings.append(func_res)
            errors.append(func_err)
        tmp = rankings[-1]
        l = len(tmp)
        for i,t in enumerate(tmp):
            tmp[i] = l-t-1
        rankings[-1] = tmp
        errors[-1] = self.error(rankings[-1],range(len(func_res)))
        rankings = np.array(rankings)
        # if cur_user in self.user_weights:
        #     weights = self.user_weights[cur_user]
        #     res = np.dot(weights,rankings)
        #     delta_weights = self.updates(errors)
        #     self.user_weights[cur_user] = self.learning_rate * weights + (1-self.learning_rate) * delta_weights
        # else:
        #     index = errors.index(min(errors))
        #     self.user_weights[cur_user] = np.array([0]*len(self.learning_functions))
        #     self.user_weights[cur_user][index] = 1
        #     res = rankings[index]
        #     delta_weights = self.updates(errors)
        if cur_user not in self.user_weights:
            weights = self.updates(errors)
        else:
            weights = self.user_weights[cur_user]
        res = np.dot(weights,rankings)
        delta_weights = self.updates(errors)
        self.user_weights[cur_user] = (self.learning_rate * weights + delta_weights)/(1+self.learning_rate)
        return res
    
    def hierarchical_OWA():
        return

    def error(self, res_pred, res_true):
        dist = 0
        for rp, rt in zip(res_pred, res_true):
            dist += (rp- rt)**2
        return dist/(len(res_pred)**2)

    def run(self, chat):
        err = []
        err_by_date = []
        for i in range(len(self.learning_functions) + 1):
            err.append([])
            err_by_date.append(defaultdict(list))
        loop = 0
        for e_i in chat:
            
            chat_e_i = defaultdict(list)
            self.follow_up_time = {}
            self.reply_time = {}
            prev_time = parser.parse(chat[e_i][0][3])
            one_day_activity = []
            for i, content in enumerate(chat[e_i]):
                # pre calculated
                if loop % 1000 == 0:
                    print(loop)
                loop += 1
                e_j = content[0]
                cur_time = parser.parse(content[3])
                chat_e_i[e_j].append(content)
                chat_history = chat_e_i[e_j]
                if content[2] == 1:
                    self.follow_up_time[e_j] = cur_time
                else:
                    self.reply_time[e_j] = cur_time
                if cur_time.date() != prev_time.date() and len(one_day_activity) > 0:      
                    # calculate the error
                    tmp_err = []
                    for func in self.learning_functions:
                        func_res = func(one_day_activity)
                        func_err = self.error(func_res,range(len(func_res)))
                        tmp_err.append(func_err)
                    # MRAC model
                    func_res = self.our_model(one_day_activity, e_j)
                    func_err =  self.error(func_res,range(len(func_res)))
                    tmp_err.append(func_err)
                    # NN model

                    # all errors
                    for k in range(len(tmp_err)):
                        err[k].append(tmp_err[k])
                        err_by_date[k][prev_time.date()].append(tmp_err[k])
                    one_day_activity = []
                    prev_time = cur_time
                # begin model
                if content[2] == 1:
                    closeness = self.closeness(chat_history, 'enron' in e_j)
                    timeliness = self.timeliness(content)
                    verbosity = content[4]
                    if e_j in self.reply_time:
                        reply_time = self.reply_time[e_j]
                    else:
                        reply_time = cur_time
                    if e_j in self.follow_up_time:
                        follow_up_time = self.follow_up_time[e_j]
                    else:
                        follow_up_time = cur_time
                    one_day_activity.append([closeness,timeliness,verbosity, follow_up_time, reply_time])
                    # deal with follow_up_time
        return err, err_by_date

path = '../data/'
idx_figure = "3"
def draw_hist_seaborn(d):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    s = ['OWA1', 'OWA2', 'baseline','our']
    for i in range(len(d)):
        ax = sns.kdeplot(d[i],label= s[i],shade = True)
    ax.set_xlim(0,1)

    ax.set(title = 'Loss')
    plt.legend()
    plt.savefig(path + "figure/enron/new/accuracy_all" + idx_figure + ".png")
    plt.close()

def draw_curve_seaborn(d):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    s = ['OWA1', 'OWA2', 'baseline','our']
    for i in range(len(d)):
        ax = plt.plot(d[i],label= s[i],alpha = 0.4)
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    plt.legend()
    plt.savefig(path + "figure/enron/new/accuracy_all_curve" + idx_figure +  ".png")
    plt.close()

def main():
    f = open(path + "chat.json",'r')
    chat = json.load(f)
    time_window_closeness = timedelta(hours = 24)
    time_window_timeliness = 0.5
    owa_parameter = 2.5
    # weights, alphas, time_window_closeness, time_window_timeliness, owa_parameter, distance_measure="two_path_num", k = 3):
    model = Model([1,1],[1,1], time_window_closeness, time_window_timeliness, owa_parameter)
    err, err_by_date = model.run(chat)
    with open("test_error", "wb") as fp:   #Pickling
        pickle.dump(err, fp)
    with open("test_error_by_date", "wb") as fp:   #Pickling
        pickle.dump(err_by_date, fp)
    error_by_date_sum = []
    keys = sorted(err_by_date[0].keys())
    for i in range(len(err_by_date)):
        tmp_error_sum = []
        for key in keys:
            tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
        error_by_date_sum.append(tmp_error_sum)
    with open("test_error_by_date_sum", "wb") as fp:   #Pickling
        pickle.dump(error_by_date_sum, fp)
    draw_hist_seaborn(err)
    draw_curve_seaborn(error_by_date_sum)

if __name__ == '__main__':
    main()