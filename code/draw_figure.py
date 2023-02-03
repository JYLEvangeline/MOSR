import argparse
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("-max_d", "-md", type=int, default= 10, help="Int parameter")
    parser.add_argument("-distance_measure", "-dm", type = str, default = "two_path_num")
    parser.add_argument("-version","-v", type = int, default = 0)
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0) # The part to delete
    parser.add_argument("-learning_rate","-lr", type = float, default = 0.99)
    parser.add_argument("-year","-y", type = int, default = 0) # 0 is for all, it could be 1999, 2000, 2001, 200
    args = parser.parse_args()
    return args


# global
args = parse()
file_names = ['all', 'part', 'all_down', 'part_down']
# check file version and down_sampling match:
if args.down_sampling != 0 and args.version not in [2, 3]:
    print("Down sampling is in version 2 and 3")
    sys.exit(1)
if args.down_sampling == 0 and args.version not in [0, 1]:
    print("Non down sampling is in version 0 and 1")
    sys.exit(1)
# initialize filename
path = '../data/'
file_version = file_names[args.version] 
if args.version in [2,3]: # down_sampling
    file_version += '_' + str(args.down_sampling)
max_d = 'max_d' + str(args.max_d) # 0 10 50 99
distance_measure = args.distance_measure
window_size = 3
idx_figure = file_version + '_' + max_d + '_' + distance_measure
if args.year != 0:
    idx_figure = "distance_" + str(args.year) +"_num_version0_max_d10_two_path_num_learning_rate0.99"
model = "enron"
colors = sns.color_palette("colorblind")
if model != 'enron':
    if args.version == 0:
        len_i = 226
    if args.version == 1:
        len_i = 126
else:
    if args.version == 0:
        len_i = 1161
    if args.version == 1:
        len_i = 247
if args.version == 0:
        len_i = 226
if args.version == 1:
    len_i = 126
if model != 'enron': # make colors united
    colors = colors[-2:] + colors[0:]
def draw_violin_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']):
    all_d = []
    all_s = []
    for dd, ss in zip(d,s):
        all_d += dd
        all_s += [ss] * len(dd) 
    df = pd.DataFrame(list(zip(all_d, all_s)), columns =['Loss', 'Method'])
    sns.set(style="whitegrid")
    sns.violinplot(x="Loss", y="Method", data= df, scale="width", palette= p)
    filename = pre_s + "violine" + model + idx_figure + ".png"
    plt.savefig(path + "figure/enron/final/" + filename)
    plt.close()

def draw_hist_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR'], name = "accuracy_all"):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in range(len(d)):
        ax = sns.kdeplot(d[i],label= s[i],shade = True)
    ax.set_xlim(0,60)
    ax.set(title = 'Errors distribution')
    plt.legend()
    filename = pre_s + name + idx_figure + ".png"
    filename = pre_s + name + ".png"
    
    plt.close()

def draw_hist_stat_general(d, title, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    # sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density")
    for i in range(len(d)):
        tmp = d[i][d[i]!=0]
        if len(tmp)>0 and max(tmp) == 1 and min(tmp) == 1:
            tmp[0] = 0.99999
        if len(tmp) == 0:
            tmp = [0]*10000 + [0.001]
        # ax = sns.kdeplot(tmp,label= s[i],shade = True, common_norm=True)
        # ax.set(title = title)
        ax = sns.displot(tmp, label= s[i], stat="probability",kde = True)
        plt.xlim(-0.1,1.1)
        plt.legend()
        filename = pre_s + title + str(i) + idx_figure + ".png" 
        filename = pre_s + title + str(i) + ".png" 
        plt.savefig(path + "figure/enron/final/" + filename)
        plt.close()

def log(d):
    return math.log((d+0.000001))

def draw_curve_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR'], name = "NDCG"):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    new_d = []
    for dd in d:
        tmp = []
        start = 0
        for i in range(max(0,len(dd) - len_i), len(dd)):
            start += dd[i]
            ddi = start/(i+1)
            tmp.append(ddi)
        new_d.append(tmp)
    for i in range(len(d)):
        if i == len(d) - 1:
            ax = plt.plot(new_d[i],label= s[i],c = colors[i]) # emphasize MOSR
            continue
        ax = plt.plot(new_d[i],label= s[i],c = colors[i])
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    # dplt.ylim(0, 140)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('LOG(Cumulative NDCG)')
    if len_i < 200:
        plt.xticks([0, 60, 120], ['08/01/2001', '10/01/2001', '12/01/2001'])
    else:
        plt.xticks([0, 60, 120, 180, 240], ['04/01/2001', '06/01/2001', '08/01/2001', '10/01/2001', '12/01/2001'])
    filename = pre_s + name + idx_figure + ".png"
    filename = pre_s + name  + ".png"
    plt.savefig(path + "figure/enron/final/" + filename)
    plt.close()

def draw_cumulative_curve_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR'], name = "accuracy"):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    new_d = []
    for dd in d:
        tmp = []
        start = 0
        for i in range(max(0,len(dd) - len_i), len(dd)):
            start += dd[i]
            tmp.append(start)
        new_d.append(tmp)
    for i in range(len(d)):
        #if i = 2:
        if i == len(d) - 1:
            ax = plt.plot(new_d[i],label= s[i],c = colors[i]) # emphasize MOSR
            continue

        ax = plt.plot(new_d[i],label= s[i],c = colors[i])
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    # dplt.ylim(0, 140)
    if len(new_d[0]) < 200:
        plt.xticks([0, 60, 120], ['08/01/2001', '10/01/2001', '12/01/2001'])
    else:
        plt.xticks([0, 60, 120, 180, 240], ['04/01/2001', '06/01/2001', '08/01/2001', '10/01/2001', '12/01/2001'])
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Loss')
    filename = pre_s + name + "_cumulative_curve" + ".png"
    filename = pre_s + name + "_cumulative_curve" + idx_figure + ".png"
    plt.savefig(path + "figure/enron/final/" + filename)
    plt.close()

def generate_pre_s_and_s(model, max_d):
    if args.version in [0,2]:
        al = "all"
    else:
        al = 'part'
    if args.version in [0,1]:
        ds = ""
    else:
        ds = "down_sampling" + str(args.down_sampling)
    if model == "msft_lr":
        pre_s = 'msft_lr' + al + max_d + ds
        s = ['MS-LR']
    elif model == 'msft_ada':
        pre_s = 'msft_ada' + al + max_d + ds
        s = ['MS-ADA']
    elif model == 'enron':
        pre_s = "distance2_" + file_version + "_num_version" + str(args.version) + "_max_d" + str(args.max_d) + '_' + args.distance_measure
        pre_s = "window" + str(3) + "/" + pre_s
        s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']
        s = ['MOSR']
    return pre_s, s

def get_err_and_err_by_date(pre_s, OWA = False):
    with open(path + pre_s + "test_error", "rb") as fp:
        err = [pickle.load(fp)]
        if 'msft' not in pre_s:
            err = err[0]

    with open(path + pre_s + "test_error_by_date", "rb") as fp:   #Pickling
        err_by_date = [pickle.load(fp)]
        if 'msft' not in pre_s:
            err_by_date = err_by_date[0]
        error_by_date_sum = []
        keys = sorted(err_by_date[0].keys())
        if OWA == False and 'msft' not in pre_s:
            start_i = len(err_by_date) - 1
        else:
            start_i = 0
        for i in range(start_i, len(err_by_date)):
            tmp_error_sum = []
            for key in keys:
                tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
            error_by_date_sum.append(tmp_error_sum)
    return err, error_by_date_sum

def get_ndcg_and_ndcg_by_date(pre_s, OWA = False):
    if 'msft' not in pre_s:
        add_learning_rate = "_learning_rate" + str(args.learning_rate)
    else:
        add_learning_rate = ""
    with open(path + pre_s + add_learning_rate + "ndcgs", "rb") as fp:
        err = [pickle.load(fp)]
        if 'msft' not in pre_s:
            err = err[0]

    with open(path + pre_s + add_learning_rate + "ndcg_by_date", "rb") as fp:
        err_by_date = [pickle.load(fp)]
        if 'msft' not in pre_s:
            err_by_date = err_by_date[0]
        error_by_date_sum = []
        keys = sorted(err_by_date[0].keys())
        if OWA == False and 'msft' not in pre_s:
            start_i = len(err_by_date) - 1
        else:
            start_i = 0
        for i in range(start_i, len(err_by_date)):
            tmp_error_sum = []
            for key in keys:
                tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
                # tmp_error_sum.append(err_by_date[i][key])
            error_by_date_sum.append(tmp_error_sum)
    return err, error_by_date_sum

def get_ndcgs(idx_figure):
    with open(path + "window" + str(window_size) + "/" + idx_figure + "ndcgs","rb") as fp:
        ndcgs = pickle.load(fp)
    return [sum(n)/len(n) for n in ndcgs]

ndcgs_all = []
if args.year != 0:
    for year in [1999, 2000, 2001, 2002]:
        idx_figure_tmp = "distance_" + str(year) +"_num_version0_max_d10_two_path_num_learning_rate0.99"
        ndcgs = get_ndcgs(idx_figure_tmp)
        ndcgs_all.append(ndcgs)
if model == "msft_lr" or model == 'msft_ada' or model == 'enron':
    pre_s, s = generate_pre_s_and_s(model, max_d)
else:
    pre_s0, s0 = generate_pre_s_and_s('msft_lr', max_d)
    pre_s1, s1 = generate_pre_s_and_s('msft_ada', max_d)
    pre_s2, s2 = generate_pre_s_and_s('enron',max_d)
    all_pre_s = [pre_s0, pre_s1,pre_s2]
    pre_s = "window" + str(window_size) + "/" + 'final' + "_num" + idx_figure
    s = s0 + s1 + s2
if model == 'msft_lr' or model == 'msft_ada':
    err, error_by_date_sum = get_err_and_err_by_date(pre_s)
    if args.max_d == 10:
        ndcg, ndcg_by_date_sum = get_ndcg_and_ndcg_by_date(pre_s)
elif model == 'enron':
    # we don't need to draw these figures now
    # with open(path + pre_s + "candidates","rb") as fp:
    #     candidates = pickle.load(fp)
    #     candidates_without_zero = []
    #     for candidate in candidates:
    #         if candidate[2] == 0:
    #             continue
    #         candidates_without_zero.append((candidate[1]-candidate[0])/candidate[2])
    #     series = ["The percentage increased after adding carbon copy receivers"]
    #     sns.kdeplot(candidates_without_zero,shade = True)
    #     filename = pre_s + "cc_list" + ".png"
    #     plt.savefig(path + "figure/enron/final/" + filename)
    #     plt.close()
    # with open(path + pre_s + "stat_true", "rb") as fp:
    #     stat_true = pickle.load(fp)
    #     stat_true = np.array(stat_true)
    #     for i in range(3):
    #         a = np.nan_to_num(stat_true[:,i]/stat_true[:,-1])
    #         stat_true[:,i] = a
    #     series = ["strangers","chat before", "have mutual connection"]
    #     draw_hist_stat_general(stat_true[:,:3].T, "Undiscovered candidates distribution num", series)
    #     for i in range(3):
    #         a = np.nan_to_num(stat_true[:,i]/stat_true[:,3])
    #         stat_true[:,i] = a
    #     draw_hist_stat_general(stat_true[:,:3].T, "Undiscovered candidates distribution", series)

    # with open(path + pre_s + "stat_pred", "rb") as fp:
    #     stat_pred = pickle.load(fp)
    #     stat_pred = np.array([stat_pred])
    #     series = ["True Positive"]
    #     draw_hist_stat_general(stat_pred, "Precision Distribution in candidates set", series)
    
    err, error_by_date_sum = get_err_and_err_by_date(pre_s)
    if args.max_d == 10:
        ndcg, ndcg_by_date_sum = get_ndcg_and_ndcg_by_date(pre_s)
else:
    err = []
    error_by_date_sum = []
    ndcg_by_date_sum = []
    for pre_s_i in all_pre_s:
        err_i, error_by_date_sum_i = get_err_and_err_by_date(pre_s_i)
        err += err_i
        error_by_date_sum += error_by_date_sum_i
        if args.max_d == 10:
            ndcg_i, ndcg_by_date_sum_i = get_ndcg_and_ndcg_by_date(pre_s_i)
            ndcg_by_date_sum += ndcg_by_date_sum_i
    l = len(err[0])
    # err = [e[-l:] for e in err]
    l = len(error_by_date_sum[0])
    error_by_date_sum = [e[-l:] for e in error_by_date_sum]
    a = [round(sum(e)/len(e),2) for e in err]
    print('max d = ' + str(max_d) + '&' + ' & '.join([str(aa) for aa in a]))
    if args.max_d == 10:
        ndcg_by_date_sum = [e[-l:] for e in ndcg_by_date_sum]

if len(error_by_date_sum) == 3:
    colors = [ "orange", "green", "royalblue"]
# Only draw violin for robustness check
# draw loss figure
if args.down_sampling != 0:
    draw_violin_seaborn(error_by_date_sum, s = s)
else:
    # draw_hist_seaborn(err, s = s)
    draw_cumulative_curve_seaborn(error_by_date_sum, s=s, name = "accuracy_msftmosr")

# draw ndcg figure
if args.max_d == 10:
    if args.down_sampling != 0:
        draw_violin_seaborn(ndcg_by_date_sum, s = s)
    else:
        draw_curve_seaborn(ndcg_by_date_sum, s=s)

# final_numpart_max_d10_two_path_numviolineallpart_max_d10_two_path_num

# final_numpart_down_0.3_max_d0_two_path_numviolineallpart_down_0.3_max_d0_two_path_num.png