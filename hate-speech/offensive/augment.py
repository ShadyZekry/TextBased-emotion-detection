from nlpaug.augmenter.word import ContextualWordEmbsAug
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pd

# BERT Augmentator
TOPK=20 #default=100
ACT = 'insert' #"substitute"

aug_bert = ContextualWordEmbsAug(
    model_path= 'UBC-NLP/MARBERT',
#     model_path='distilbert-base-uncased', 
    device='cuda',
    action=ACT, top_k=TOPK)

def augment_text(df, augmenter, label_val, samples=100, pr=0.2, show = 0):
    augmenter.aug_p = pr
    new_text=[]
    
    # selecting the minority class samples
    df_n = df[df['label']==label_val].reset_index(drop=True)

    # data augmentation loop
    for i in tqdm(np.random.randint(0, len(df_n), samples)):
            text = df_n.iloc[i]['tweet']
            augmented_text = augmenter.augment(text)
            if show:
                print(f"The original text: {text}\nThe new one: {augmented_text}" )
                print('-'*50)
            new_text.append(augmented_text)
    
    # dataframe
    new = pd.DataFrame({'tweet':new_text,'label':label_val})
    df = shuffle(pd.concat([df,new]).reset_index(drop=True))
    return df


# data = pd.read_csv('./dataset/OSACT2022-sharedTask-train.csv', usecols=['tweet', 'hs_label'])

# data.loc[data['hs_label'] == 'HS1', 'label'] = 1
# data.loc[data['hs_label'] == 'HS2', 'label'] = 1
# data.loc[data['hs_label'] == 'HS3', 'label'] = 1
# data.loc[data['hs_label'] == 'HS4', 'label'] = 1
# data.loc[data['hs_label'] == 'HS5', 'label'] = 1
# data.loc[data['hs_label'] == 'HS6', 'label'] = 1
# data.loc[data['hs_label'] == 'NOT_HS', 'label'] = 0

# hs_count = data.loc[data['label'] == 1].shape[0]
# print(hs_count, ' hs count')
# not_hs_count = data.loc[data['label'] == 0].shape[0]
# print(not_hs_count, ' non hs count')


# num_samples = not_hs_count - hs_count
# aug_df = augment_text(data, aug_bert, label_val= 1, samples=num_samples)
# aug_df["tweet"] = [string.replace("[UNK]", "")  for string in aug_df['tweet']]
# aug_df['tweet'] = aug_df['tweet'].astype(str)
# aug_df.to_csv('./result/augmented_hatespeech_data.csv', index=False)

# off_data = pd.read_csv('./dataset/OSACT2022-sharedTask-train.csv', usecols=['tweet', 'off_label'])

# off_data.loc[off_data['off_label'] == 'OFF', 'label'] = 1
# off_data.loc[off_data['off_label'] == 'NOT_OFF', 'label'] = 0

# off_count = off_data.loc[off_data['label'] == 1].shape[0]
# print(off_count, ' off count')
# not_off_count = off_data.loc[off_data['label'] == 0].shape[0]
# print(not_off_count, ' non off count')

# num_samples = not_off_count - off_count
# aug_df = augment_text(off_data, aug_bert, label_val =1, samples=num_samples)
# aug_df["tweet"] = [string.replace("[UNK]", "")  for string in aug_df['tweet']]
# aug_df['tweet'] = aug_df['tweet'].astype(str)
# aug_df.to_csv('./result/augmented_offensive_data.csv', index=False)

data = pd.read_csv('./dataset/OSACT2022-sharedTask-train.csv', usecols=['tweet', 'hs_label'])

data.loc[data['hs_label'] == 'HS1', 'label'] = 1
data.loc[data['hs_label'] == 'HS2', 'label'] = 2
data.loc[data['hs_label'] == 'HS3', 'label'] = 3
data.loc[data['hs_label'] == 'HS4', 'label'] = 0
data.loc[data['hs_label'] == 'HS5', 'label'] = 4
data.loc[data['hs_label'] == 'HS6', 'label'] = 5
data.loc[data['hs_label'] == 'NOT_HS', 'label'] = 0

hs_count = data.loc[data['label'] != 0].shape[0]
print(hs_count, ' hs count')
not_hs_count = data.loc[data['label'] == 0].shape[0]
print(not_hs_count, ' non hs count')

num_samples = 1393 - data.loc[data['label'] == 1].shape[0]
hs1_aug_data = augment_text(data, aug_bert, label_val=1, samples=num_samples)
hs1_aug_data['tweet'] = [string.replace("[UNK]", "") for string in hs1_aug_data['tweet']]

num_samples = 1393 - data.loc[data['label'] == 2].shape[0]
hs2_aug_data = augment_text(hs1_aug_data, aug_bert, label_val=2, samples=num_samples)
hs2_aug_data['tweet'] = [string.replace("[UNK]", "") for string in hs2_aug_data['tweet']]

num_samples = 1393 - data.loc[data['label'] == 3].shape[0]
hs3_aug_data = augment_text(hs2_aug_data, aug_bert, label_val=3, samples=num_samples)
hs3_aug_data['tweet'] = [string.replace("[UNK]", "") for string in hs3_aug_data['tweet']]

num_samples = 1393 - data.loc[data['label'] == 4].shape[0]
hs5_aug_data = augment_text(hs3_aug_data, aug_bert, label_val=4, samples=num_samples)
hs5_aug_data['tweet'] = [string.replace("[UNK]", "") for string in hs5_aug_data['tweet']]

num_samples = 1393 - data.loc[data['label'] == 5].shape[0]
hs6_aug_data = augment_text(hs5_aug_data, aug_bert, label_val=5, samples=num_samples)
hs6_aug_data['tweet'] = [string.replace("[UNK]", "") for string in hs6_aug_data['tweet']]

hs6_aug_data.to_csv('./result/augmented_multiclass_data.csv', index=False)

