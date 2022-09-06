import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

fig, ax = plt.subplots(2, 3)
fig.suptitle('A single ax with no data')
ax[0, 0].set_title('data1')
ax[0, 1].set_title('data2')
ax[0, 2].set_title('0 2')
ax[1, 0].set_title('1 0')

data1 = pd.DataFrame(dict(name=['a', 'b'], value=[2, 3]))
data2 = pd.DataFrame(dict(name=['c', 'd'], value=[4, 5]))

sns.barplot(ax=ax[0, 0], data=data1, x='name', y='value')
sns.barplot(ax=ax[0, 1], data=data2, x='name', y='value')
plt.show()