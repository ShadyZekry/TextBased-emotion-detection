import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_dataset_df = pd.read_csv('./result/OSACT2022-taskA-train.csv', usecols=['tweet','off_label'], encoding='utf-8')

sns.countplot(x='off_label', data=train_dataset_df).set_title('Off label count plot')

plt.show()

