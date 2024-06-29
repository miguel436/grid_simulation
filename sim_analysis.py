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

plt.figure()

plt.title('Median Grid Score')

plt.plot(df['angle'], df['median'])
plt.axvline(x=16, color='red', linestyle="dotted")

plt.xlabel('Tilt Angle (degrees)')
plt.ylabel('Median score')
plt.xticks([0] + list(range(0, 48, 4)))
plt.xlim(0, 46)


plt.figure()
plt.title('Mean Grid Score')

plt.plot(df['angle'], df['mean'])
plt.axvline(x=16, color='red', linestyle="dotted")

plt.xlabel('Tilt Angle (degrees)')
plt.ylabel('Mean score')
plt.xticks([0] + list(range(0, 48, 4)))
plt.xlim(0, 46)
plt.show()
