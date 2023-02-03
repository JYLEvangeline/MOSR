import argparse
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("-max_d", "-md", type=int, default= 10, help="Int parameter")
    parser.add_argument("-distance_measure", "-dm", type = str, default = "two_path_num")
    parser.add_argument("-version","-v", type = int, default = 1)
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0.3) # The part to delete
    parser.add_argument("-learning_rate","-lr", type = float, default = 0.99)
    parser.add_argument("-year","-y", type = int, default = 2000)
    args = parser.parse_args()
    return args



# global
args = parse()
file_names = ['all', 'part', 'all_down', 'part_down']
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
model = "all"
p = sns.color_palette("colorblind")
if model == 'all':
    if args.version == 0:
        len_i = 226
    if args.version == 1:
        len_i = 126
else:
    if args.version == 0:
        len_i = 1161
    if args.verion == 1:
        len_i = 247
if model != 'enron': # make colors united
    p = p[-2:] + p[0:]
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

def draw_hist_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    for i in range(len(d)):
        ax = sns.kdeplot(d[i],label= s[i],shade = True)
    ax.set_xlim(0,60)
    ax.set(title = 'Errors distribution')
    plt.legend()
    filename = pre_s + "accuracy_all" + idx_figure + ".png"
    filename = pre_s + "accuracy_all" + ".png"
    
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

def draw_curve_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    new_d = []
    for dd in d:
        tmp = []
        start = 0
        for i in range(len(dd)):
            start += dd[i]
            tmp.append(dd[i]/(i+1))
        new_d.append(tmp)
    for i in range(len(d)):
        #if i != 2:
        ax = plt.plot(new_d[i],label= s[i],alpha = 0.4)
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    # dplt.ylim(0, 140)
    plt.legend()
    filename = pre_s + "accuracy_all_curve" + idx_figure + ".png"
    filename = pre_s + "accuracy_all_curve"  + ".png"
    plt.savefig(path + "figure/enron/final/" + filename)
    plt.close()

def draw_cumulative_curve_seaborn(d, s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    new_d = []
    for dd in d:
        tmp = []
        start = 0
        for i in range(len(dd) - len_i, len(dd)):
            start += dd[i]
            tmp.append(start)
        new_d.append(tmp)
    for i in range(len(d)):
        #if i = 2:
        if i == len(d) - 1:
            ax = plt.plot(new_d[i],label= s[i],alpha = 0.4, c = p[i], linewidth=5.5)
            continue

        ax = plt.plot(new_d[i],label= s[i],alpha = 0.4, c = p[i])
    # ax.set_xlim(0,60)
    # ax.title(title = 'Errors curve')
    # dplt.ylim(0, 140)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Loss')
    filename = pre_s + "accuracy_cumulative_curve" + ".png"
    filename = pre_s + "accuracy_cumulative_curve" + idx_figure + ".png"
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
        s = ['msft_lr']
    elif model == 'msft_ada':
        pre_s = 'msft_ada' + al + max_d + ds
        s = ['msft_ada']
    elif model == 'enron':
        pre_s = "distance2_" + file_version + "_num_version" + str(args.version) + "_max_d" + str(args.max_d) + '_' + args.distance_measure
        pre_s = "window" + str(3) + "/" + pre_s
        s = ['OWA1', 'OWA2','OWA3','OWA4','timeline','MOSR']
    return pre_s, s

def get_err_and_err_by_date(pre_s):
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
        for i in range(len(err_by_date)):
            tmp_error_sum = []
            for key in keys:
                tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
            error_by_date_sum.append(tmp_error_sum)
    return err, error_by_date_sum

def get_ndcgs(idx_figure):
    with open(path + "window" + str(window_size) + "/" + idx_figure + "ndcgs","rb") as fp:
        ndcgs = pickle.load(fp)
    return [sum(n)/len(n) for n in ndcgs]

def draw(dic, fig_name):
    year1 = ["1999to2001", "2000to2001", "2001to2001"]
    year2 = ["2001to1999", "2001to2000", "2001to2001"]
    for key in year1:
        value = dic[key]
        plt.plot(range(6), value, label = "train at " + key[:4])
    plt.legend()
    plt.title("test at 2001")
    plt.xticks(range(6),['OWA1', 'OWA2', 'OWA3', 'OWA4','baseline','MOSR'])
    plt.savefig(fig_name + "train_different.jpg")
    plt.close()
    for key in year2:
        value = dic[key]
        plt.plot(range(6), value, label = "test at " + key[-4:])
    plt.legend()
    plt.title("train at 2001")
    plt.xticks(range(6),['OWA1', 'OWA2', 'OWA3', 'OWA4','baseline','MOSR'])
    plt.savefig(fig_name + "test_different.jpg")
    plt.close()


path = '../data/window3/'
pre_name  =  "distance2_all_num_version0_max_d10_two_path_num_learning_rate0.99"
options = ["train", "test"]
years = ["1999to2001", "2000to2001", "2001to1999", "2001to2000", "2001to2001"]

for option in ['test','test_without_change']:
    dic_loss = {}
    dic_ndcgs = {}
    for year in years:
        with open( path + pre_name + option + year, "rb") as fp:   #Pickling
            data  = pickle.load(fp)
            loss = [sum(d)/len(d) for d in data[-3]]
            ndcgs = [sum(d)/len(d) for d in data[-1]]
            dic_loss[year] = loss
            dic_ndcgs[year] = ndcgs
    draw(dic_loss, '../data/figure/enron/final/year_comparison/' + option + "loss")
    draw(dic_ndcgs, '../data/figure/enron/final/year_comparison/' + option + "ndcgs")


# draw_violin_seaborn(error_by_date_sum, s = s)
# draw_hist_seaborn(err, s = s)
# draw_cumulative_curve_seaborn(error_by_date_sum, s=s)
