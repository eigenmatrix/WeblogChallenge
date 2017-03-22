import math
from matplotlib import pyplot as plt

def plot_requests(file_name):
    x =[]
    y =[]
    with open(file_name, 'r') as f:
        for l in f:
            a,b = eval(l.strip())
            # if a % 3 ==1:
            x.append(a)
            y.append(b)
    plt.scatter(x, y)
    plt.show()

if __name__ == '__main__':
    plot_requests("../data/seconds_request/combined.txt")