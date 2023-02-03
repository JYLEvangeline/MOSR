import math
import matplotlib.pyplot as plt
import random
def var(x1, x2):
    delta_x1 = x2
    delta_x2 = math.sin(x1) - x2
    return x1 + delta_x1, x2+ delta_x2

def one_loop():
    x1 = (random.random() - 0.5) * 6 * math.pi
    x2 = (random.random() - 0.5) * 4
    data = [[x1], [x2]]
    for i in range(100):
        x1, x2 = var(x1, x2)
        data[0].append(x1)
        data[1].append(x2)
    return data

random.seed(1)
for i in range(5):
    data = one_loop()
    plt.plot(data[0], data[1])
plt.show()