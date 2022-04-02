import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_dataset_df = pd.read_csv('./result/OSACT2022-taskA-train.csv', usecols=['tweet','hs_label'], encoding='utf-8')

sns.countplot(x='hs_label', data=train_dataset_df).set_title('Hate speach label count plot')

plt.show()

