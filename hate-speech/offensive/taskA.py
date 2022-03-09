import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from BertUtilities import BertUtilities
from ModelsUtilities import ModelsUtilities
from TfIdfUtilities import TfidfUtilities
from sklearn.utils.class_weight import compute_class_weight
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

task_name = 'taskA'
train_preprocessed_path = './result/OSACT2022-sharedTask-train.csv'
val_preprocessed_path = './result/OSACT2022-sharedTask-dev.csv'
marbert_model_path = './marbert-model/'
task_dataset_cols = ['tweet','off_label']
tweet_col_index = 0
label_col_index = 1
imbalance_handlers = ['none', 'weighted-classes']

batch_size = 32
num_classes = 2
num_epochs = 200

models_utils = ModelsUtilities(task_name)

train_df = pd.read_csv(train_preprocessed_path, usecols=task_dataset_cols).dropna()
val_df = pd.read_csv(val_preprocessed_path, usecols=task_dataset_cols).dropna()

# train_df = train_df[(train_df[task_dataset_cols[label_col_index]] == 'OFF') | (train_df[task_dataset_cols[label_col_index]] == 'NOT_OFF')]
# val_df = val_df[(val_df[task_dataset_cols[label_col_index]] == 'OFF') | (val_df[task_dataset_cols[label_col_index]] == 'NOT_OFF')]

train_df.loc[train_df[task_dataset_cols[label_col_index]] == 'OFF', task_dataset_cols[label_col_index]] = 1
train_df.loc[train_df[task_dataset_cols[label_col_index]] == 'NOT_OFF', task_dataset_cols[label_col_index]] = 0
train_df[task_dataset_cols[label_col_index]] = train_df[task_dataset_cols[label_col_index]].astype(np.float64)

val_df.loc[val_df[task_dataset_cols[label_col_index]] == 'OFF', task_dataset_cols[label_col_index]] = 1
val_df.loc[val_df[task_dataset_cols[label_col_index]] == 'NOT_OFF', task_dataset_cols[label_col_index]] = 0
val_df[task_dataset_cols[label_col_index]] = val_df[task_dataset_cols[label_col_index]].astype(np.float64)

train_df = train_df.sample(frac=1)

train_tfidf = TfidfUtilities(train_df, task_dataset_cols[tweet_col_index], 300, .5, 2, (1,1), None)
train_features_df = train_tfidf.extract_features()

val_tfidf = TfidfUtilities(val_df, task_dataset_cols[tweet_col_index], 300, .5, 2, (1,1), train_tfidf.get_vocab())
val_features_df = val_tfidf.extract_features()

x_train_tfidf = train_features_df.values
y_train = to_categorical(train_df[task_dataset_cols[label_col_index]].values, num_classes=num_classes)

x_val_tfidf = val_features_df.values
y_val = to_categorical(val_df[task_dataset_cols[label_col_index]].values, num_classes=num_classes)

class_weights = compute_class_weight('balanced', classes=np.unique(train_df[task_dataset_cols[label_col_index]]), y=train_df[task_dataset_cols[label_col_index]])
print(f'computed class weights: {class_weights}')
# models_utils.build_and_fit_gru(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn_gru(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[0], True)

# models_utils.build_and_fit_gru_weighted(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_weighted(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_gru_weighted(x_train_tfidf, y_train, x_val_tfidf, y_val, batch_size, num_epochs, models_utils.feature_methods[0], imbalance_handlers[1], class_weights, True)

train_marbert_utils = BertUtilities(task_name, 'train', marbert_model_path, marbert_model_path, train_df, task_dataset_cols[tweet_col_index])
x_train_embds = train_marbert_utils.read_embeddings_from_disk()
if type(x_train_embds) == type(None):
    train_marbert_utils.tokenize_dataset()
    train_marbert_utils.forward_to_bert()
    x_train_embds = train_marbert_utils.get_embeddings_holder()

val_marbert_utils = BertUtilities(task_name, 'val', marbert_model_path, marbert_model_path, val_df, task_dataset_cols[tweet_col_index])
x_val_embds = val_marbert_utils.read_embeddings_from_disk()
if type(x_val_embds) == type(None):
    val_marbert_utils.tokenize_dataset()
    val_marbert_utils.forward_to_bert()
    x_val_embds = val_marbert_utils.get_embeddings_holder()

# models_utils.build_and_fit_gru(x_train_embds, y_train, x_val_embds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn(x_train_embds, y_train, x_val_embds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[0], True)
# models_utils.build_and_fit_cnn_gru(x_train_embds, y_train, x_val_embds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[0], True)

# models_utils.build_and_fit_gru_weighted(x_train_embds, y_train, x_val_embds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_weighted(x_train_embds, y_train, x_val_embds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[1], class_weights, True)
# models_utils.build_and_fit_cnn_gru_weighted(x_train_embds, y_train, x_val_emds, y_val, batch_size, num_epochs, models_utils.feature_methods[1], imbalance_handlers[1], class_weights, True)

