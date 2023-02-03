
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statistics
import json
import seaborn as sns

from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def mean(user_loss):
    avg = []
    for val in user_loss.values():
        if len(val)!=0:
            avg.append(sum(val)/len(val))
    return statistics.mean(avg)
train_name = '2001.6.1'
test_names = ['2001.10.1', '2001.11.1', '2001.12.1', '2002.1.1', '2002.2.1']
# non_vs_stat = [[3596, 20595], [6993, 12499], [5863, 6785], [4751, 5083], [3435, 3371]] # val2
# non_vs_stat = [[9490, 14701], [7935, 11557], [5263, 7385], [4223, 5611], [3060, 3746]] # val1
path = 'data/'
def generate_x_and_y(res_dic, k = 2, round_decimal = 2, maxi_loss = 999, option = 'MOSR', ndcg = False):
    x = []
    y = []
    dic = defaultdict(list)
    for key in res_dic.keys():
        if ndcg == False:
            try:
                val = statistics.mean([statistics.mean(avg)/len(avg)**k if (len(avg) > 5) else maxi_loss for avg in res_dic[key]]) # MOSR
            except:
                try:
                    val = statistics.mean(res_dic[key]) # MSFT
                except:
                    val = maxi_loss
        else:
            try:
                val = statistics.mean([statistics.mean(avg)  if avg != [] else 0 for avg in res_dic[key]]) # MOSR
            except:
                try:
                    val = statistics.mean(res_dic[key]) # MSFT
                except:
                    val = 0
        # val = statistics.mean([statistics.mean(avg)/len(avg)  if avg != [] else 0 for avg in res_dic[key]])
        x.append(round(float(key),round_decimal))
        y.append(val)
        dic[round(float(key),round_decimal)].append(val)
    for key in dic.keys():
        dic[key] = statistics.mean(dic[key])
    return list(dic.keys()), list(dic.values())
    return x, y


def draw_length_vs_loss(res_dic, k = 3):
    x = []
    y = []
    dic = defaultdict(list)
    for key in res_dic.keys():
        for avg in res_dic[key]:
            if avg != []:
                try:
                    val = statistics.mean(avg)
                except:
                    print(1)
                x.append(len(avg)**k)
                y.append(val)
                dic[len(avg)**k].append(val)
    for key in dic.keys():
        dic[key] = statistics.mean(dic[key])
    x = list(dic.keys())
    y = list(dic.values())
    return x, y

def get_bin_average(xys):
    # cut the bin to 0-2,2-4,4-6,6-8
    new_xys = []
    for xy in xys:
        ys = []
        for i in range(4):
            new_y = xy[1][i*20:(i+1)*20]
            ys.append(round(statistics.mean(new_y),4))
        new_xys.append(ys)
    for new_xy in new_xys:
        s = [str(xy) for xy in new_xy]
        print("& ".join(s))
    return new_xys
        


def MOSR(file_path =  path + 'break_into_consistent/MOSR_results3seperate_val1', ndcg = False, including_baselines = True):
    # file_path = path + 'break_into_consistent/MOSR_results'
    # [0, 1, 2, 3, 4, 5] -> ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'baseline','our']
    if ndcg == False:
        idx = 3
    else:
        idx = 4
    xys = []
    if including_baselines == False: # whether to show ['OWA1', 'OWA2', 'OWA3', 'OWA4' 'timeline']
        start_i = 5
    else:
        start_i = 0
    json_diff_dics = {}
    for i in range(start_i, 6):
        without_change_res_dic = defaultdict(list)
        with_change_res_dic = defaultdict(list)
        dic = pickle.load(open(file_path,"rb"))
        for test_name in test_names:
            without_change_results  = dic[test_name + '_without_change'][idx] # idx = 3 is loss, idx = 4 is ndcg
            with_change_results  = dic[test_name + '_with_change'][idx]
            
            # read json_diff_dics to make it faster
            vs_path = path + 'break_into_consistent/MOSR3/' + train_name + 'vs' + test_name + str(120) +".json"
            if vs_path not in json_diff_dics:  
                json_diff_dic = json.load(open(vs_path,"r"))
                json_diff_dics[vs_path] = json_diff_dic
            json_diff_dic = json_diff_dics[vs_path]
            for name in without_change_results.keys():
                if name in json_diff_dic:
                    key = str(round(json_diff_dic[name], 1))
                    without_change_res_dic[key].append(without_change_results[name][i])
                    with_change_res_dic[key].append(with_change_results[name][i])
        if i == 5:
            x, y = draw_length_vs_loss(without_change_res_dic)
            plt.scatter(x,y)
            plt.xlabel("length**2")
            plt.ylabel("loss")
            plt.savefig(path + "figure/enron/final/length**2_vs_loss.png", bbox_inches='tight')
            plt.close()
        x, y = generate_x_and_y(with_change_res_dic, ndcg = ndcg)
        y = [yy for _, yy in sorted(zip(x,y))]
        x = sorted(x)
        xys.append([x,y])
    if including_baselines == True:
        get_bin_average(xys)
    return xys


def msft(file_path =  path + 'break_into_consistent/msft_ada_result/', maxi_loss = 999, k = 2, ndcg = False):

    if ndcg == False:
        idx = 2
    else:
        idx = 3
    # file_path = path + 'break_into_consistent/MOSR_results'
    res_dic = defaultdict(list)
    # dic = pickle.load(open(file_path,"rb"))
    for test_name in test_names:
        dic = pickle.load(open(file_path + train_name + 'vs' + test_name+ 'seperate_val1','rb'))
        # res_dic  = dic[test_name + '_without_change'][3]
        vs_path = path + 'break_into_consistent/MOSR3/' + train_name + 'vs' + test_name + str(120) +".json"
        json_diff_dic = json.load(open(vs_path,"r"))
        for name in dic[idx].keys(): # idx = 2 is loss, idx = 3 is ndcg
            if name in json_diff_dic:
                key = str(round(json_diff_dic[name], 2))
                if ndcg == False:
                    if dic[idx][name][1] > 5:
                        res_dic[key].append(dic[idx][name][0]/dic[idx][name][1]**k)
                    else:
                        res_dic[key].append(maxi_loss)
                else:
                    if dic[idx][name][1] != 0:
                        res_dic[key].append(dic[idx][name][0])
                    else:
                        res_dic[key].append(0)
    # for key in res_dic.keys():
    #     sum_of_val = 0
    #     for val in res_dic[key]:
    #         if val != 'NA':
    #             sum_of_val += val
    #         else:
    #             print(1)
    # x, y = draw_length_vs_loss(res_dic)
    # plt.scatter(x,y)
    # plt.xlabel("length**2")
    # plt.ylabel("loss")
    # plt.savefig(path + "figure/enron/final/length**2_vs_loss.png", bbox_inches='tight')
    # plt.close()
    x, y = generate_x_and_y(res_dic, option = 'msft')
    return x, y
    # plt.scatter(x,y)
    # plt.xlabel("corr")
    # plt.ylabel("loss")
    # plt.savefig(path + "figure/enron/final/corr_vs_loss.png", bbox_inches='tight')
    # plt.close()

def draw_loss(all_xy, labels = ['MS-LR', 'MS-ADA', 'OWA1', 'OWA2', 'OWA3', 'OWA4','timeline','MOSR']):
    if len(labels) == 8:
        colors = sns.color_palette("colorblind")
    else:
        colors = [ "orange", "green", "royalblue"]
    fig, ax = plt.subplots(1, 1)
    for i in range(len(all_xy)):
        x, y = all_xy[i]
        label_xy = labels[i]
        ax.scatter(x,y, label = label_xy, color = colors[i])
    ax.set_xlabel(r'Non-stationary behavior coefficient $\tau$')
    ax.set_ylabel("Loss")
    ax.set_ylim(-1, 2000)
    ax.legend()
    # zoom in
    locs = ["upper right", "center right", "lower right"]
    for i in range(len(all_xy)):
        axins = inset_axes(ax, width="40%", height="30%",loc=locs[i],
                   bbox_to_anchor=(0.6, 0, 1, 1),
                   bbox_transform=ax.transAxes)
        if i == 1 or i == 0:
            axins.set_ylim(-1,2000)
        
        x, y = all_xy[i]
        label_xy = labels[i]
        axins.scatter(x, y, label = label_xy, c = [colors[i]])

    
    plt.savefig(path + "figure/enron/final/corr_vs_loss.png", bbox_inches='tight')   
    plt.close()


def draw_ndcg(all_xy, labels = ['MS-LR', 'MS-ADA', 'OWA1', 'OWA2', 'OWA3', 'OWA4','timeline','MOSR']):
    if len(labels) == 8:
        colors = sns.color_palette("colorblind")
    else:
        colors = [ "orange", "green", "royalblue"]
    fig, ax = plt.subplots(1, 1)
    for i in range(len(all_xy)):
        x, y = all_xy[i]
        label_xy = labels[i]
        ax.scatter(x,y, label = label_xy, color = colors[i])
    ax.set_xlabel(r'Non-stationary behavior coefficient $\tau$')
    ax.set_ylabel("NDCG")
    ax.legend()
    # zoom in
    locs = ["upper right", "center right", "lower right"]
    
    for i in range(len(all_xy)):
        axins = inset_axes(ax, width="40%", height="30%",loc=locs[i],
                   bbox_to_anchor=(0.6, 0, 1, 1),
                   bbox_transform=ax.transAxes)
        # if i == 1 or i == 2:
        #     axins.set_ylim(-1, 50000)
        
        x, y = all_xy[i]
        label_xy = labels[i]
        axins.scatter(x, y, label = label_xy, c = [colors[i]])

    
    plt.savefig(path + "figure/enron/final/corr_vs_ndcg.png", bbox_inches='tight')   
    plt.close()


# draw loss
all_xy = []
xy_mosr = MOSR(ndcg = False, including_baselines = False)
xy_msftlr = msft(file_path =  path + 'break_into_consistent/msft_lr_result/wrong/', ndcg = False)
xy_msftada = msft(file_path =  path + 'break_into_consistent/msft_ada_result/wrong/', ndcg = False)
all_xy.append(xy_msftlr)
all_xy.append(xy_msftada)
all_xy += xy_mosr
draw_loss(all_xy, ['MS-LR', 'MS-ADA','MOSR'])

# # draw ndcgs
all_xy = []
xy_mosr = MOSR(ndcg = True, including_baselines = False)
xy_msftlr = msft(file_path =  path + 'break_into_consistent/msft_lr_result/', ndcg = True)
xy_msftada = msft(file_path =  path + 'break_into_consistent/msft_ada_result/', ndcg = True)
all_xy.append(xy_msftlr)
all_xy.append(xy_msftada)
all_xy += xy_mosr
draw_ndcg(all_xy, ['MS-LR', 'MS-ADA','MOSR'])


# print OWA
xy_mosr = MOSR(ndcg = False, including_baselines = True)
xy_mosr = MOSR(ndcg = True, including_baselines = True)
