from datetime import datetime, timedelta


import json

# the consistence of reply email
def main(pre_s, idx_figure):
    f = open(path + file_name,'r')
    chat = json.load(f)
    time_window_closeness = timedelta(hours = 24)
    time_window_timeliness = 0.5
    owa_parameter = 2.5
    # weights, alphas, time_window_closeness, time_window_timeliness, owa_parameter, distance_measure="two_path_num", k = 3):
    model = Model([1,1],[1,1], time_window_closeness, time_window_timeliness, owa_parameter)

if __name__ == '__main__':
    path = '../data/'
    file_name = "chat_distance2.json"
    idx_figure = "1"
    pre_s = file_name[5:-5]
    main(pre_s, idx_figure)