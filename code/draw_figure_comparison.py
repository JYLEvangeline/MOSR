# draw hyper parameters

import argparse
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import pandas as pd




def parse():
    parser = argparse.ArgumentParser(description='add, modify and delete upstream nodes')
    parser.add_argument("-max_d", "-md", type=int, default= 10, help="Int parameter")
    parser.add_argument("-distance_measure", "-dm", type = str, default = "two_path_num")
    parser.add_argument("-version","-v", type = int, default = 1)
    parser.add_argument("-down_sampling", "-ds", type = float, default= 0.3) # The part to delete
    parser.add_argument("-learning_rate","-lr", type = float, default = 0.99)
    args = parser.parse_args()
    return args

args = parse()
file_names = ['all', 'part', 'all_down', 'part_down']
# initialize filename
path = '../data/'


def draw_multiple(data, axi, hypers = ['two_path_num', 'shortest_path']):
    sns.set_theme(style="white")
    # label = [0,0.5,1,2]
    sns.set(color_codes = True)
    if len(hypers) == 2:
        column_hyper = "Diatance Measure"
    else:
        column_hyper = "Learning Rate"
    # make data a df
    df = []
    s = ['OWA1', 'OWA2','OWA3','OWA4','MOSR']
    for data_i, hyper in zip(data, hypers):
        for data_j, ss in zip(data_i, s):
            df.append([data_j, hyper, ss])
    df = pd.DataFrame(df, columns=['Loss', column_hyper, 'Methods'])
    sns.barplot(ax = axi, x = column_hyper, y = "Loss", hue = 'Methods', data = df)
    # plt.legend(loc='upper left')
    # filename = 'comparison' + column_hyper + ".png"
    # plt.savefig(path + "figure/enron/final/" + filename)
    # plt.close()
    #ax.set_xlim([0, 10])
    

def read_error_by_data(pre_s):
    with open(path + pre_s + "test_error_by_date", "rb") as fp:   #Pickling
        err_by_date = [pickle.load(fp)]
        if 'msft' not in pre_s:
            err_by_date = err_by_date[0]
    error_by_date_sum = []
    keys = sorted(err_by_date[0].keys())
    avg = []
    for i in range(len(err_by_date)): # i is algorithm
        mean = 0
        l = 0
        tmp_error_sum = []
        for key in keys: #key is the date
            tmp_error_sum.append(sum(err_by_date[i][key])/len(err_by_date[i][key]))
            mean += sum(err_by_date[i][key])
            l += len(err_by_date[i][key])
        error_by_date_sum.append(sum(tmp_error_sum)/len(tmp_error_sum))
        avg.append(round(mean/l,2))
    return error_by_date_sum, avg

def draw_hyperparameter(axi, type = "learning_rate"):
    file_version = file_names[args.version] 
    if args.version in [2,3]: # down_sampling
        file_version += '_' + str(args.down_sampling)
    # if type == 'learning_rate':
    max_d = 'max_d' + str(args.max_d) # 0 10 50 99
    distance_measure = args.distance_measure
    window_size = 3
    idx_figure = file_version + '_' + max_d + '_' + distance_measure
    model = "enron"
    pre_s = "distance2_" + file_version + "_num_version" + str(args.version) + "_max_d" + str(args.max_d) + '_' 
    errors = []
    avgs = []
    if type == "distance":
        for distance_measure in ['two_path_num', 'shortest_path']:
            dis_pre_s = pre_s + distance_measure
            file_name = "window" + str(3) + "/" + dis_pre_s
            error, avg = read_error_by_data(file_name )
            errors.append(error[:-2]+error[-1:]) # ignore time
            avgs.append(avg[:-2]+avg[-1:])
            hypers = ['two_path_num', 'shortest_path']
    if type == "learning_rate":
        dis_pre_s = pre_s + distance_measure
        file_name = "window" + str(3) + "/" + dis_pre_s
        for lr in [0.5, 0.8, 0.9]:
            error, avg = read_error_by_data(file_name + '_learning_rate' + str(lr))
            errors.append(error[:-2]+error[-1:])
            avgs.append(avg[:-2]+avg[-1:])
        # lr is 0.99
        error, avg = read_error_by_data(file_name)
        errors.append(error[:-2]+error[-1:]) # ignore time
        avgs.append(avg[:-2]+avg[-1:])
        hypers =  [0.5, 0.8, 0.9, 0.99]
    print(type)
    print(avgs)
    draw_multiple(errors, axi, hypers = hypers)

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

draw_hyperparameter(ax1, type = "learning_rate")
draw_hyperparameter(ax2, type = "distance")
ax1.legend_.remove()
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.legend(loc = 2, bbox_to_anchor = (1.05,1.0), borderaxespad = 0)
# plt.show()


plt.savefig(path + "figure/enron/final/hyper.png", bbox_inches='tight')
plt.close()