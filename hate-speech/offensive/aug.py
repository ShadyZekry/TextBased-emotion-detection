from nlpaug.augmenter.word import ContextualWordEmbsAug
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pd

# BERT Augmentator
TOPK=20 #default=100
ACT = 'insert' #"substitute"

data = pd.read_csv('./result/OSACT2022-sharedTask-train.csv', usecols=['tweet', 'hs_label'])

data.loc[data['hs_label'] == 'HS1', 'label'] = 1
data.loc[data['hs_label'] == 'HS2', 'label'] = 1
data.loc[data['hs_label'] == 'HS3', 'label'] = 1
data.loc[data['hs_label'] == 'HS4', 'label'] = 1
data.loc[data['hs_label'] == 'HS5', 'label'] = 1
data.loc[data['hs_label'] == 'HS6', 'label'] = 1
data.loc[data['hs_label'] == 'NOT_HS', 'label'] = 0

hs_count = data.loc[data['label'] == 1].shape[0]
print(hs_count, ' hs count')
not_hs_count = data.loc[data['label'] == 0].shape[0]
print(not_hs_count, ' non hs count')

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

aug_df = augment_text(data, aug_bert, label_val= 1, samples=5*10**3)

aug_df.to_csv('./result/augmented_hatespeech_data.csv', index=False)

# augmantated_data = augment_text(aug_data, aug_bert, label= 'sarcastic', samples=1600)