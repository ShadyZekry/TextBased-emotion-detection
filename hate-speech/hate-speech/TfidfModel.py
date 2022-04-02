import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from HandleImbalances import use_smote
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

train_dataset_df = pd.read_csv('./Result/OSACT2022-taskA-train.csv', usecols=['tweet','off_label'], encoding='utf-8')
eval_dataset_df = pd.read_csv('./Result/OSACT2022-taskA-dev.csv', usecols=['tweet','off_label'], encoding='utf-8')

tfidf_vectorizer = TfidfVectorizer(lowercase=False, encoding='utf-8', input='content', ngram_range=(1,2), token_pattern=r"(?u)\b\w\w+\b", min_df=2, max_df=0.5, max_features=300)
features = tfidf_vectorizer.fit_transform(train_dataset_df['tweet'])

features_df = pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names())

x_train = features_df.values
y_train = to_categorical(train_dataset_df['off_label'], num_classes=2)

x_train, y_train = shuffle(x_train, y_train)

eval_tfidf_vectorizer = TfidfVectorizer(lowercase=False, encoding='utf-8', input='content', ngram_range=(1,2), token_pattern=r"(?u)\b\w\w+\b", min_df=2, max_df=0.5, vocabulary=tfidf_vectorizer.get_feature_names())
eval_features = eval_tfidf_vectorizer.fit_transform(eval_dataset_df['tweet'])
eval_features_df = pd.DataFrame(eval_features.todense(), columns=tfidf_vectorizer.get_feature_names())

x_eval = eval_features_df.values
y_eval = to_categorical(eval_dataset_df['off_label'], num_classes=2)

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(x_train.shape[1], activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=5e-5), loss='binary_crossentropy', metrics=[Precision(), Recall(), AUC()])

model_checkpoint_callback = ModelCheckpoint(
    filepath='./tfidf_temp/checkpoint',
    save_weights_only=True,
    monitor='val_precision',
    mode='max',
    save_best_only=True)

model.fit(x_train, y_train,validation_data=(x_eval, y_eval), batch_size=32, epochs=10, verbose=2, callbacks=model_checkpoint_callback)

model.load_weights('./tfidf_temp/checkpoint')

print("Saving model and weights started")
model_json = model.to_json()
with open('./tfidf_model/model.json', 'w') as file:
    file.write(model_json)

model.save_weights('./tfidf_model/weights.h5')
print("Saving model and weights ended")

#tfidf model with smote over resampling technique for class imbalancing

x_train_resampled, y_train_resampled = use_smote(x_train, y_train) 
y_train_resampled = to_categorical(y_train_resampled, num_classes = 2)

model_checkpoint_callback = ModelCheckpoint(
    filepath='./tfidf_resampled_temp/checkpoint',
    save_weights_only=True,
    monitor='val_precision',
    mode='max',
    save_best_only=True)

print('Training on resampled data started')

model.fit(x_train_resampled, y_train_resampled,validation_data=(x_eval, y_eval), batch_size=32, epochs=10, verbose=2, callbacks=model_checkpoint_callback)

print('Training on resampled data ended')

model.load_weights('./tfidf_resampled_temp/checkpoint')

print("Saving model and weights started")
model_json = model.to_json()
with open('./tfidf_resampled_model/model.json', 'w') as file:
    file.write(model_json)

model.save_weights('./tfidf_resampled_model/weights.h5')
print("Saving model and weights ended")



