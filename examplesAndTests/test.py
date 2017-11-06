import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


SEQ_LEN = 10

sns.set(color_codes=True)
def create_time_series():
    freq = (np.random.random() * 0.5) + 0.1  # 0.1 to 0.6
    ampl = np.random.random() + 0.5  # 0.5 to 1.5
    x = np.sin(np.arange(0, SEQ_LEN) * freq) * ampl
    return x


for i in range(0, 5):
    sns.tsplot(create_time_series())  # 5 series

plt.show()


def to_csv(filename, N):
    with open(filename, 'w') as ofp:
        for lineno in range(0, N):
            seq = create_time_series()
            line = ",".join(map(str, seq))
            ofp.write(line + '\n')


to_csv('train.csv', 1000)  # 1000 sequences
to_csv('valid.csv', 50)
