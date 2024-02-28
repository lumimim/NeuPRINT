import matplotlib.pyplot as plt

def plot_histogram(lst):
    plt.figure(figsize=(5,2))
    plt.hist(lst, bins=range(len(set(lst))+1))
    plt.xticks(range(len(set(lst))))
    plt.show()