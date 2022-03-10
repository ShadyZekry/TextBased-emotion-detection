import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.utils import to_categorical
from ModelsUtilities import ModelsUtilities
from sklearn.utils.class_weight import compute_class_weight
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

task_name = 'taskA'
train_preprocessed_path = './result/OSACT2022-sharedTask-train.csv'
val_preprocessed_path = './result/OSACT2022-sharedTask-dev.csv'
marbert_model_path = './marbert-model/'
task_dataset_cols = ['tweet','off_label']
tweet_col_index = 0
label_col_index = 1
imbalance_handlers = ['none', 'weighted-classes']
bert_model_name = 'marbert'

batch_size = 64
num_classes = 2
num_epochs = 2000

marbert_model = TFAutoModel.from_pretrained(marbert_model_path, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(marbert_model_path, from_tf=True)

models_utils = ModelsUtilities(task_name, marbert_model, tokenizer)

train_df = pd.read_csv(train_preprocessed_path, usecols=task_dataset_cols).dropna()
val_df = pd.read_csv(val_preprocessed_path, usecols=task_dataset_cols).dropna()

train_df.loc[train_df[task_dataset_cols[label_col_index]] == 'OFF', task_dataset_cols[label_col_index]] = 1
train_df.loc[train_df[task_dataset_cols[label_col_index]] == 'NOT_OFF', task_dataset_cols[label_col_index]] = 0
train_df[task_dataset_cols[label_col_index]] = train_df[task_dataset_cols[label_col_index]].astype(np.float64)

val_df.loc[val_df[task_dataset_cols[label_col_index]] == 'OFF', task_dataset_cols[label_col_index]] = 1
val_df.loc[val_df[task_dataset_cols[label_col_index]] == 'NOT_OFF', task_dataset_cols[label_col_index]] = 0
val_df[task_dataset_cols[label_col_index]] = val_df[task_dataset_cols[label_col_index]].astype(np.float64)

train_df = train_df.sample(frac=1)

y_train = to_categorical(train_df[task_dataset_cols[label_col_index]].values, num_classes=num_classes)

y_val = to_categorical(val_df[task_dataset_cols[label_col_index]].values, num_classes=num_classes)

class_weights = compute_class_weight('balanced', classes=np.unique(train_df[task_dataset_cols[label_col_index]]), y=train_df[task_dataset_cols[label_col_index]])
print(f'computed class weights: {class_weights}')

# models_utils.build_and_fit_gru(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn_gru(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[0], True)


models_utils.build_and_fit_gru_weighted(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_weighted(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_gru_weighted(train_df[task_dataset_cols[tweet_col_index]], y_train, val_df[task_dataset_cols[tweet_col_index]], y_val, batch_size, num_epochs,f'{bert_model_name}_embeddings', imbalance_handlers[1], class_weights, True)

