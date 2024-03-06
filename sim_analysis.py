import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import normaltest


def fix_scores(scores):
    true_scores = scores[::2]
    reverse_scores = true_scores[::-1]

    final_scores = np.sum([true_scores, reverse_scores], axis=0)
    return final_scores

df = pd.read_csv('./simulation_long_tiago.csv')
df['scores'] = df['scores'].apply(eval)

df['normality'] = df['scores'].apply(normaltest)


"""
df.drop(columns='Unnamed: 0', inplace=True)

df['scores'] = df['scores'].apply(fix_scores)

df['mean'] = df['scores'].apply(np.mean)
df['median'] = df['scores'].apply(np.median)
df['stdev'] = df['scores'].apply(np.std)
df = df.head(-1)
"""

"""
fig, ax = plt.subplots(nrows=3, ncols=1)

ax[0].plot(df['angle'], df['mean'])
ax[1].plot(df['angle'], df['median'])
ax[2].plot(df['angle'], df['stdev'])

for i in range(0, 3, 1):
    ax[i].axvline(x=8, color='red')
    ax[i].axvline(x=16, color='red')


plt.show()
"""
