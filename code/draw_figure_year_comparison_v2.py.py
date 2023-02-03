import pickle
import matplotlib.pyplot as plt

from collections import defaultdict
def average(l):
    return sum(l)/len(l)

def draw(average_losses, ndcgs, test_ys, train_y, train_length):
    xticks = ['MS-LR', 'MS-ADA', 'OWA1', 'OWA2', 'OWA3', 'OWA4','timeline','MOSR-v1','MOSR-v2']
    for value, test_y in zip(average_losses, test_ys):
        plt.plot(range(len(value)), value, label = "test at " + test_y)
    plt.legend()
    plt.title("train at " + train_y + " " + str(train_length) + " days")
    plt.xticks(range(len(value)),xticks)
    plt.savefig(path + train_y + str(train_length) + "average_losses.jpg")
    plt.close()
    for value, test_y in zip(ndcgs, test_ys):
        plt.plot(range(len(value)), value, label = "test at " + test_y)
    plt.legend()
    plt.title("train at " + train_y + " " + str(train_length) + " days")
    plt.xticks(range(len(value)),xticks)
    plt.savefig(path + train_y + str(train_length) + "ndcgs.jpg")
    plt.close()

test_ys = ['1999.1.1', '2000.1.1','2001.1.1']
train_ys = ['2001.1.1','2001.9.1','2001.10.1','2001.11.1']
train_lengths = [180]

path = '../data/differently/'
# msft
names_msft = []
methods = ["lr", "ada"]
for method in methods:
    names_msft.append("msft/msft_" + method + "allmax_d10")
# our
names_our = []
methods = ["test", "test_without_change"]
for method in methods:
    names_our.append("our/distance2_all_num_version0_max_d10_two_path_num_learning_rate0.99" + method) 

all_average_loss = defaultdict(list)
all_ndcgs = defaultdict(list)
all_names = defaultdict(list)
for train_length in train_lengths:
    for train_y in train_ys:
        for test_y in test_ys:
            average_loss = []
            ndcgs = []
            for name in names_msft:
                try:
                    with open( path + name + 'test' + train_y + 'to' + test_y + 'train_length' + str(train_length), "rb") as fp:   #Pickling
                        tmp_data  = pickle.load(fp)
                        average_loss.append(average(tmp_data[0]))
                        ndcgs.append(average(tmp_data[2]))
                except:
                    with open( path + name + train_y + 'to' + test_y + "rb") as fp:   #Pickling
                        tmp_data  = pickle.load(fp)
            for name in names_our:
                with open( path + name + train_y + 'to' + test_y + 'train_length' + str(train_length), "rb") as fp:   #Pickling
                    tmp_data  = pickle.load(fp)
                    for d in tmp_data[0]:
                        average_loss.append(average(d))
                    for d in tmp_data[2]:
                        ndcgs.append(average(d))
            average_loss = average_loss[:8] + average_loss[-1:]
            ndcgs = ndcgs[:8] + ndcgs[-1:]
            all_average_loss[train_y].append(average_loss)
            all_ndcgs[train_y].append(ndcgs)
            all_names[train_y].append(test_y)
        
    for key in all_average_loss.keys():
        draw(all_average_loss[key], all_ndcgs[key], all_names[key], key, train_length)