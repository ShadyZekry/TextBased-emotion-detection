from ModelsUtilities import ModelsUtilities
from utilities import preprocess
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.activations import softmax
import numpy as np
from tqdm import tqdm

offensive_task_name = 'taskA'
hate_task_name = 'taskB'
marbert_model_path = './marbert-model/'
column_names = ['id','tweet','off_label', 'hs_label']
id_col_index = 0
tweet_col_index = 1
off_label_index = 2
hate_label_index = 3
num_classes = 2 
batch_size = 64

imbalance_handlers = ['none', 'weighted-classes']
bert_model_name = 'marbert'
marbert_model = TFAutoModel.from_pretrained(marbert_model_path, output_hidden_states=True)
marbert_tokenizer = AutoTokenizer.from_pretrained(marbert_model_path, from_tf=True)

# hs_val_df = pd.read_csv('./result/OSACT2022-sharedTask-dev.csv', usecols=column_names).dropna()
hs_val2_df = pd.read_csv('./result/L-HSAB-train.csv', usecols=['tweet', 'class']).dropna()
hs_val3_df = pd.read_csv('./result/L_HSAB-dev.csv', usecols=['tweet', 'class']).dropna()

# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS1', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS2', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS3', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS4', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS5', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'HS6', column_names[hate_label_index]] = 1
# hs_val_df.loc[hs_val_df[column_names[hate_label_index]] == 'NOT_HS', column_names[hate_label_index]] = 0
# hs_val_df[column_names[hate_label_index]] = hs_val_df[column_names[hate_label_index]].astype(np.float64)

hs_val2_df.loc[hs_val2_df['class'] != 'hate', 'class'] = 0
hs_val2_df.loc[hs_val2_df['class'] == 'hate', 'class'] = 1
hs_val2_df['class'] = hs_val2_df['class'].astype(np.float64)

hs_val3_df.loc[hs_val3_df['class'] != 'hate', 'class'] = 0
hs_val3_df.loc[hs_val3_df['class'] == 'hate', 'class'] = 1
hs_val3_df['class'] = hs_val3_df['class'].astype(np.float64)

# y_val1 = to_categorical(hs_val_df[column_names[hate_label_index]], num_classes = num_classes)
y_val2 = to_categorical(hs_val2_df['class'], num_classes = num_classes)
y_val3 = to_categorical(hs_val3_df['class'], num_classes = num_classes)

y_val = np.concatenate([y_val2, y_val3], axis=0)
x_val = np.concatenate([hs_val2_df['tweet'].values, hs_val3_df['tweet'].values], axis=0)

for i in tqdm(range(x_val.shape[0]), desc='Preprocessing Leventine dataset'):
    x_val[i] = preprocess(x_val[i])

hate_model_utils = ModelsUtilities(hate_task_name, marbert_model, marbert_tokenizer)

hate_cnn_model = hate_model_utils.read_model(hate_model_utils.models_names[1], f'{bert_model_name}_embeddings', imbalance_handlers[1])

hate_model_utils.evaluate(hate_cnn_model, x_val, y_val, batch_size, True)

