import pandas as pd
from tqdm import tqdm
from utilities import preprocess

testing_dataset_df = pd.read_csv('./dataset/OSACT2022-sharedTask-dev.csv',  usecols=['id','tweet', 'off_label', 'hs_label'],encoding='utf-8')
training_dataset_df = pd.read_csv('./dataset/OSACT2022-sharedTask-train.csv',  usecols=['id','tweet', 'off_label', 'hs_label'],encoding='utf-8')

for index in tqdm(range(training_dataset_df.index.shape[0]), desc='Preprocessing training dataset..'):
    training_dataset_df['tweet'].at[index] = preprocess(training_dataset_df['tweet'].at[index])

training_dataset_df.to_csv('./result/OSACT2022-sharedTask-train.csv', columns=['id','tweet', 'off_label', 'hs_label'], index=False)

for index in tqdm(range(testing_dataset_df.index.shape[0]),desc='Preprocessing test dataset..'):
    testing_dataset_df['tweet'].at[index] = preprocess(testing_dataset_df['tweet'].at[index])

testing_dataset_df.to_csv('./result/OSACT2022-sharedTask-dev.csv', columns=['id','tweet', 'off_label', 'hs_label'], index=False)








